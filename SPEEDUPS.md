# SPEEDUPS: MD inference performance changes on branch `warp-moment-kernel`

Review document for the responsible engineers. One entry per change: **motivation** (what profiling showed), **change**
(what and where), **verification** (against which reference), **effect** (measured), **risks / limits**.
Detailed profiles, HLO excerpts and failed experiments: `../perf_notes.md` (lab notebook) and `../HANDOFF.md`.

Base: `model-and-perf-improvements` @ `6b5c046`. Hardware for all numbers: NVIDIA L40S (Ada, sm_89), JAX 0.10.2, warp-lang
1.19. Benchmark: ecemc EC:EMC/LiPF6 GMNN (r_max 6, 12 Bessel, 12 radial / 5 tensor channels, packed moments, NLH),
NVT 330 K, dt 1 fs, 3,861 and 17,875 atoms; steady-state median of 500-step chunks (spread ~ +/-4 %).
Scripts: `../ecemc/warp_bench/` (`bench.sh`, `nsys_arms.sh`, `top_kernels.py`, `check_*.py`, `ab_forces.py`).

**Overall** (recommended config: `moment_kernel: warp`, `contract_kernel: warp`, `neighbor_grad: pull`,
`fuse_nlh: true`, `capacity_multiplier: 1.05`, `neighbor_builder: alchemi` (+ default `sort_atoms: true`), default
`matmul_precision: highest`) vs original apax tabulated-radial MD (TF32), on scrambled atom order (the state of a long
run; ordered start structures are within 0.6 %): **4.5x at 3.9k atoms (2160 vs 485 steps/s, 187 ns/day), 8.7x at
17.9k atoms (763 vs 87.6 steps/s, 66 ns/day)**; 104k atoms 99 steps/s (8.6 ns/day). Measured 2026-11-24, L40S.
Accuracy vs a **full-fp64 model** (descriptor + readout fp64, exact radial, fp64 geometry; `check_fp64.py`), 3.9k
cubic and sheared triclinic boxes: **MD forces** (fp32 geometry, what drives the dynamics) max 0.15-0.19 meV/A, rms
0.021 meV/A, the same as the XLA fp32 path (0.16-0.19 / 0.023-0.024): the residual is fp32 position/displacement
rounding, shared by all arms. Trajectory/auxiliary forces (fp64 geometry) max 0.008-0.012, rms 0.002 meV/A. Energy
5-7e-4 eV per 3,861 atoms (fp32 round-off). NVE (dt 0.5 fs, full stack vs original XLA path): total energy std 8.7 vs 8.3
meV, drift -5.2 vs -4.5 ueV/atom/ps (equivalent; `../perf_notes.md` V1). The original TF32 setup gave up to 32 meV/A.

Accuracy policy used throughout: every change is checked against an fp64 reference (forces max/rms, energy delta) and
must stay within forces < 1-5 meV/A, energy ~ fp32 round-off. Reduced-precision tricks (TF32, fast_math) were rejected.

---

## A. Opt-in fast path (MD config)

### A1. Warp Gaussian-moment kernels: `moment_kernel: warp`
* Motivation: the XLA descriptor materialises several pair-sized arrays (radial, compressed radial, R1 x dn^k: ~120
  floats per pair) and scatters them with 4 `segment_sum`s: ~600 us of GPU time per force call at 3.9k atoms.
* Change: `apax/layers/descriptor/warp_moments.py` (new). Forward: atom-centric (thread per atom x radial channel x
  `n_split`), loops the atom's contiguous neighbour segment, Bessel basis via one sincos + Chebyshev recurrence,
  per-pair-type table (`TabulatedRadialFunction`) with the radial compression folded into a second table, accumulates
  m0 and packed m1/m2/m3 in registers. Backward: pair-parallel, writes dE/d(dr_vec), no atomics. `jax.custom_vjp` with
  explicit array arguments. Wiring: `GaussianMomentDescriptor.moment_kernel`, `GMNNBuilder`, MDConfig, `md_setup`.
* Verification: moments and VJP vs fp64 XLA on a real 561k-pair list: rms rel 1.6e-7 / 4.0e-7 (XLA fp32: 1.2e-4 / 2.1e-4).
  NVE drift (3.9k, real masses, 20 ps): -0.0004 meV/atom/ps vs -0.0012 for XLA.
* Effect: 1.45x (3.9k) / 1.96x (17.9k) on its own (session 1).
* Limits: tabulated Bessel radial with identity transform only; fp32; no hessians; inference only; forward atomics
  (`n_split > 1`) make results non-bitwise-reproducible (~1e-7). Needs `warp-lang` (set `WARP_CACHE_PATH`).

### A2. Scatter-free neighbour-position gradient: `neighbor_grad: pull`
* Motivation: the backward of `R[idx[0]]` (neighbour gather) is an unsorted atomic scatter-add; each atom is the
  neighbour of ~100 others, so it was the single most expensive kernel of every apax MD step (0.4 ms of 1.5 ms at
  3.9k, 1.9 ms at 17.9k).
* Change: `apax/layers/neighbor_grad.py` (new). jax-md `Sparse` lists are symmetric and centre-sorted, so the neighbour
  sum equals a gather from each pair's reverse slot over the atom's own segment. `gather_pair(R, idx)` (custom VJP
  for both position gathers): backward = Warp reverse-pair search + Warp segmented reduction
  `dR[b] = sum_{q in seg(b)} ct_c[q] + ct_n[rev q]` (fp32; fp64 cotangents use the exact take + segment_sum).
  Wiring: `make_distance_fn(neighbor_grad=...)`, `EnergyModel.neighbor_grad`, builder, MDConfig.
* Verification: vs the original scatter: dE 0, max dF 3e-6 eV/A, max dstress 4e-6.
* Effect: +10-15 % alone; larger combined with A1 (session 1 + E3a/E3b below).
* Limits: jax-md `Sparse` lists (symmetric, centre-sorted, no duplicate pairs: guaranteed by apax's minimum-image check);
  MD only, not training batches.

### A3. `matmul_precision: highest` (default) -- session 2, E4
* Motivation: on Ampere+ GPUs fp32 dots/einsums run in TF32 by default (~1e-3 relative). Measured force error from the
  remaining TF32 einsums (readout, compression, contractions): max 1.7 meV/A, rms 0.15 meV/A.
* Change: MDConfig `matmul_precision` (default "highest") sets `jax_default_matmul_precision` in `run_md` before tracing.
* Verification: forces vs fp64 geometry: max 0.20, rms 0.023 meV/A; energy ~1e-3 eV on 3,861 atoms (fp32 round-off).
  Against a full-fp64 model (2026-11-24, `check_fp64.py`) every fp32 arm with "highest" is within 0.013 meV/A max;
  the same script without "highest" (TF32) gives 1.6-1.8 meV/A max, 0.36-0.40 rms.
* Effect: -3.9 % (3.9k), -0.6 % (17.9k) speed. Set "default" to get TF32 back.

### A5. `fuse_nlh: true` (opt-in) -- session 2, E8 (see table C)

### A4. `capacity_multiplier` (default 1.25 = unchanged) -- session 2, E6
* Motivation: jax-md's sparse list capacity = occupancy x 1.25 + N x `extra_capacity` (per atom for Sparse): at 17.9k,
  2.6M slots for 1.85M pairs. Every pair-level kernel runs over all slots.
* Change: MDConfig `capacity_multiplier`, passed to `partition.neighbor_list`.
* Effect at 1.05: +3.4 % (3.9k), +7 % (17.9k). 60 ps NVT: 0 overflows even at 1.0.
* Risks: an overflow costs a reallocation + recompile + redone chunk (now logged). Recommended 1.05 for fixed-volume NVT
  production; keep the default for NPT / heating / unequilibrated starts.

---

## B. Automatic changes (no config; affect every periodic apax MD run)

### B1. Elementwise periodic displacement -- session 1
* Motivation: jax-md's `transform` is `einsum("ij,...j->...i", box, dR)`. XLA lowers it as a Triton GEMM over millions of
  3-vectors: slow (1.37 ms of each ~2.3 ms neighbour-list rebuild, plus the per-step pair displacement) and **TF32**:
  force errors up to 32 meV/A (rms 6.7) vs fp64 on a triclinic box.
* Change: `apax/utils/space.py::periodic_general_elementwise` (box product as explicit multiply-adds; jax-md's custom
  JVP kept; explicit `box=` keeps jax-md's dtype promotion; the captured default box follows the positions' dtype so
  fp32 neighbour lists stay fp32; scalar / diagonal boxes fall back). Used by `md_setup`.
* Verification: parity with jax-md (values, d/dR, d/dbox, perturbation; triclinic, mixed dtypes) 1e-7 fp32 / 1e-15 fp64;
  end-to-end forces vs fp64 improved 40x.
* Effect: +11 % (XLA path), +25 % (Warp path); +4-5 % more from keeping the neighbour list in fp32.

### B2. No per-step `inv(box)` in constant-box ensembles -- session 1
* Motivation: jax-md's `shift_fn` recomputes `inv(box)` (cuSOLVER LU + triangular solves) whenever called with `box=`,
  which the integrator does every step.
* Change: `md_setup` drops `box=` from `shift_fn` calls for non-NPT ensembles (precomputed inverse is used).
* Effect: ~3 %. NPT unchanged.

### B3. Logging / profiling hooks -- session 1-2
* `run_sim` logs each chunk's wall time and every neighbour-list overflow redo (info level).
* `APAX_PROFILE_CHUNKS=K` / `APAX_PROFILE_SKIP=n`: `cudaProfilerStart/Stop` around steady chunks for
  `nsys --capture-range=cudaProfilerApi`. No effect unless set.

---

## C. Session-2 refinements inside the fast path (all exact; numbers are Warp + pull at the time of the change)

| id | change (file) | motivation (kernel, us/step at 3.9k) | verification | effect 3.9k / 17.9k |
|---|---|---|---|---|
| E1 | Unpack packed symmetric moments with static slices (`moments.unpack_symmetric`) on the Warp path | `p3[..., _G3]` gather -> its transpose is a scatter-add: `input_scatter_fusion` f32[N,5,10] 23 us + f32[N,5,6] 7 us | values bit-equal, VJP 9e-16 | +2.7 % / +3.5 % |
| E2 | CSR segment starts by a Warp kernel (`neighbor_grad._starts_kernel`) | `segment_sum(ones)` + cumsum: int scatter over all pairs, 18.5 us | bit-equal on real + edge-case lists (`check_csr.py`) | +2.3 % / +1 % |
| E3a | One custom VJP for both position gathers (`gather_pair`) | two sorted segment sums (46 + 32 us) | pull vs scatter 3e-6 eV/A | +4.2 % / +4.7 % |
| E3b | Warp segmented reduction for that backward (`_reduce_kernel`) | XLA take+add+sorted segment_sum fusion, 50.7 us | 3e-6 eV/A; fp64 path kept exact | +4.8 % / +1 % |
| E8 | Fused NLH repulsion (`fuse_nlh: true`, opt-in, needs `moment_kernel: warp`): NLH gradient added inside `gm_backward` (one `dr_vec` consumer, no pair-cotangent `add_any`); NLH energy in a separate per-atom Warp kernel (`nlh_energy`) that XLA drops in force-only steps | NLH pair loop fwd+bwd ~210 us + pair-cotangent add 118 us at 17.9k (DRAM-bound) | vs unfused: max dF 1.1e-5 eV/A, dstress 3.6e-6 rel, dE 6.7e-4 eV (< fp32 round-off of the total) | +1.6 % / **+7.2 %** |
| E10 | Reverse-pair search only from pairs with centre < neighbour, writing `rev[p]` and `rev[q]` (involution; one writer per slot; pre-filled -1) | `reverse_pairs` 162 us at 17.9k (41 us at 3.9k) | 0 mismatches vs brute force on the real list; forces unchanged | reverse kernel 162 -> 118.5 us; +1 % / +0.5 % |
| E11 | `gm_backward`: m0 part via a per-(atom, neighbour element) table `G0[a,e,n] = sum_k T0[type,n,k] g[a,k]` (new tiny kernel) instead of 144 FMAs + 12 loads per pair | `gm_backward` register-limited (112 regs, 33 % occupancy), 231 us at 17.9k | VJP vs fp64 identical (4.0e-7) | `gm_backward` 231 -> 152 us; +1.2 % / +1.5 % |
| E12 | Neighbour list built by NVIDIA ALCHEMI Toolkit-Ops (`neighbor_builder: alchemi`, opt-in; `pip install nvalchemi-toolkit-ops`, base package only: numpy + warp-lang). apax keeps only a thin jax-md-compatible adapter (`apax/md/alchemi_neighbor_list.py`: allocate/sizing, skin rebuild criterion, `NeighborList` + overflow flags); the search kernels are NVIDIA-maintained | jax-md's cell-list rebuild materialises 31M candidate pairs at 17.9k (~1,750/atom for ~100 neighbours): ~300 us/step amortised | identical pair sets vs jax-md on cubic 3.9k / 17.9k and sheared triclinic boxes; same capacity; rebuild criterion checked (`check_nl.py`) | +1.6 % / **+12 %**; peak GPU memory 639 vs 763 MiB (3.9k), 1571 vs 1537 MiB (17.9k) |
| E13 | `gm_forward` threads regrouped: one thread per (atom, **tensor** channel c, split) doing channel c's tensor moments plus scalar channels c, c+C, ... (3/3/2/2/2 of 12), instead of one per scalar channel; split heuristic target 2^17 threads | basis recomputed 12x per pair; channels >= C mostly idle (imbalance); forward 338 us at 17.9k | moments/VJP vs fp64 unchanged (1.6e-7 / 4.0e-7) | +1.5 % / **+4.0 %** |
| E14 | Spatial (Morton) atom sorting at each neighbour-list rebuild (`sort_atoms`, **default on** with `neighbor_builder: alchemi`; `alchemi_neighbor_list.py`: `morton_order`, `SortedNeighborList`, `sorted_frame`). The list is built in sorted index space; the model is evaluated on `R[perm]` and per-atom outputs mapped back, so integrator / trajectory / constraints keep the input order. The permutation keeps the species pattern (`Z[perm] == Z`), so the model's compile-time-constant Z is unchanged | MD diffusion scrambles any spatial order of atom indices within ~1 ns; a randomly ordered input is 9.0 % slower at 17.9k and 7.4 % at 104k than a spatially ordered one (gathers / atomics lose cache locality). Benchmarks start from ordered structures, so long runs were ~7-9 % slower than quoted | sorted vs unsorted: energy 1e-5 meV/atom, forces at fp32 round-off; same distance from fp64 (`check_sort.py`, `check_fp64.py`) | before E15: 104k scrambled 40.1 -> 43.0 (+7.5 %), but 17.9k -5 to -13 % (schedule flip, note below). **On top of E15** (scrambled = long-run state): 3.9k 1702 -> 1745 (+2.5 %), **17.9k 565 -> 640 (+13 %)**, **104k 67.8 -> 76.5 (+13 %)**; on ordered input -3 % (1811 -> 1756, 668 -> 649) |
| E15 | Gaussian-moment contractions in Warp (`contract_kernel: warp`, opt-in; `layers/descriptor/warp_contract.py`): forward one thread per (atom, feature) from the packed moments (no unpacking); hand-derived VJP as 7 family-specialised kernels over a static (channel, family, slot) work list, one thread per (chunk, atom), component-major (transposed) operands so each warp access is one coalesced line, 3 / 6 / 10 atomics per thread; feature order from the same `tril_*_indices` / `n_contr` / `drop_contractions` as `_contract` | XLA contraction: 325 us fwd + 793 us bwd + 196 us unpacking = **1.31 ms of a 2.43 ms step** at 17.9k (54 %), memory-bound and L2/schedule-sensitive | kernel vs fp64 `_contract`: fwd 1e-7, VJP 1e-7 rel (`check_contract.py`, incl. drop/n_contr variants); end-to-end forces vs full fp64 0.013 meV/A max (cubic + triclinic); NVE conservation as XLA path | **+30 % / +68 %** (1399 -> 1821, 396 -> 666 steps/s); contraction 1.31 ms -> ~0.36 ms (fwd 105 us, bwd ~250 us) |
| E16 | Reverse-pair map computed once per neighbour-list rebuild instead of every step: the ALCHEMI adapter stores it as a third row of `idx` (`_with_reverse`); `neighbor_grad.reverse_pairs` uses that row when present (jax-md lists unchanged). All other apax code reads only `idx[0]` / `idx[1]` | `reverse_pairs` kernel 105 us/step at 17.9k (7 % of the step after E15), recomputed although the list changes only every ~8 steps | `check_nl.py`: (nbr, centre)[rev[p]] == (centre, nbr)[p] for every real pair, P for padding, on 3.9k / 17.9k / triclinic; forces vs fp64 unchanged | **+6 % / +7-8 %** (3.9k 1745 -> 1852, 17.9k 640 -> 693 scrambled; 104k +1.6 %). Session 2 tried a cache with jax-md lists: +2-3 %, not kept (needed loop state) |
| E19 | E7 again, now kept: all Warp kernels take the whole (rows, P) pair array and index rows themselves (`pairs[0, p]`, `pairs[1, p]`, reverse slots from row 2 of the cached array) instead of row slices | nsys memcpy trace: 4 x 8.9 MB DtoD copies/step (32 us) from row-slice custom-call operands, plus 2 x 26.7 MB around the update cond (19 us) and `memcpy128` 36 us | check_csr / check_rev / check_moments / check_nl / check_fp64 (jax-md and alchemi lists, triclinic) unchanged | +0.7 % / **+3.6 %** (1852 -> 1865, 691 -> 716); 104k +2.8 %. The session-2 loss (below) was a side effect on XLA's contraction kernels, gone with E15 |
| E21 | Fused periodic pair vectors (`neighbor_grad.pair_vectors`, custom VJP, 2 Warp kernels): forward `dr = box @ (s - floor(s + 1/2))`, `s = R[centre] - R[nbr]` straight from fractional positions; backward `dR[b] = sum_{q in seg(b)} ct[q] - ct[rev[q]]` (fractional tangents pass through unscaled, the apax/jax-md convention). Used by `compute_distances` for `neighbor_grad: pull` + fractional minimum-image displacement + constant box (tags set by `periodic_general_elementwise` and `md_setup` for non-NPT ensembles); stress (perturbation) and NPT take the general path | ~150 us/step at 17.9k of pair-sized XLA work: two gathers, `vmap()/add` f32[P,3], `remainder` and-fusion, `slice` f32[P,1], transpose neg/add f32[P,3], plus the separate centre/neighbour-cotangent reduce. Session 2's E5 (same idea, no gain) predates the NLH fusion: now `dr_vec` has one consumer | `check_pairvec.py`: dr within 2.4e-6 A (fp32 wrap rounding), VJP bit-identical, jax-md and ALCHEMI lists, cubic and triclinic. MD forces vs full fp64: max 0.153 / rms 0.0213 meV/A (XLA path 0.163 / 0.0239) | **+5 % / +5 %**, **104k +22 %** (1865 -> 1963, 716 -> 753, 79.9 -> 97.8); kernels 1327 -> 1235 us/step at 17.9k. First attempt keyed on "box is not a tracer": never true inside the integrator (the box is traced even when constant), caught by profiling |
| E23 | Nose-Hoover chain: 1/Q formed once per half step, substeps multiply instead of divide (`md/nhc_fused.py`) | the fused chain is one serial dependency chain of ~60 fp64 `exp` and ~70 fp64 divisions per step, latency-bound on the 1/64-rate fp64 units: ~38 us/step regardless of N (7.5 % of a 3.9k step) | `check_nhc.py`: one half step from identical states vs the previous code: scale-factor error 1.2e-8 relative to \|scale - 1\|, p_xi 7.8e-10 relative (fp64 reassociation). Closed-loop 100k half steps diverge by ~2e-4 in p_xi for *any* 1-ulp change (chain sensitivity), KE agrees to 1e-9 | **+3 % / +0.3 %** (1963 -> 2022, 753 -> 755); chain fusions 38 -> 23 us/step. Variant with fp32 `expm1` for the scale factors: +0.7 % more, 10x the (still tiny) per-step error: not kept |
| E24 | Forward split heuristic `default_split`: S = 2^round(log2 sqrt(3e5 / (N C))) instead of "smallest S with N C S >= 2^17" | after E13-E21 the 3.9k optimum moved: S sweep 2 / 4 / 8 / 16 / 32 -> 2032 / **2085** / 2035 (old default) / 1876 / 1544 steps/s; 17.9k S 1 / 2 / 4 -> 733 / **757** / 744; 104k S 1 / 2 -> 97.8 / 98.3. The optimum thread count grows ~sqrt(N) (77k threads at 3.9k, 179k at 17.9k), no fixed target fits both | exact (launch shape only) | **+3.9 %** at 3.9k (2022 -> 2101), 17.9k / 104k unchanged (same S) |
| E25 | Contraction backward: families 6 and 7 get one kernel per slot position (`make_bwd(FAM, POS)`, accumulator kind static), and the chunk length scales with N (`chunk_length`: 4 at 3.9k, 16 at 17.9k, 32 from ~64k) | ncu 17.9k: the family-6/7 kernels (95 / 105 us) used 166 / 126 registers (16 / 31 % occupancy): all three position branches' D-tensor temporaries live in one kernel. Splitting alone made 3.9k 7.7 % slower: each (family, position) grid is n_chunks x N threads, ~45 blocks at 3.9k | `check_contract.py` unchanged (fwd 1e-7, VJP 0.8-1.2e-7) | **+2.8 % / +0.6 %** (2101 -> 2160, 759 -> 763), 104k +1.3 %; 17.9k backward 243 -> 233 us. Chunk sweep: 3.9k 2 / 4 / 8 / 16 / 32 -> 2140 / 2154 / 2110 / 2029 / 1940 steps/s; 17.9k 4-64 flat; 104k 32 beats 4 by 1.7 % |
| E7 (reverted) | Pass the whole (2, P) pair array to Warp kernels (no row slices) | a row slice as custom-call operand is materialised: 4 x 10.4 MB DtoD copies per step at 17.9k (~55 us) | exact | copies gone, but **-5.6 % at 17.9k** (A/B x2), 0 % at 3.9k: unrelated XLA kernels got slower. Kept slices; see D. |

Note on E14 at 17.9k (and E7): the sorting ops themselves are ~free (identity-permutation control is equally slow),
but any small change to the graph's inputs flipped XLA's kernel *order* in the contraction/readout region; identical
kernels (same HLO and launch config; ncu with cache flush: same 102 us / 84 MB DRAM) then ran 2.5x slower in situ
because their inputs (7 x 9 MB GEMM outputs) were no longer L2-resident. `--xla_gpu_enable_latency_hiding_scheduler`
true/false does not change it. E15 removes that XLA region; on top of E15 sorting costs only ~3 % on ordered input
and the scrambled-order penalty it removes grew to 12-15 % (faster step, locality matters more).

E15 history: v1 (one thread per (atom, channel) pulling over all features) took 6.9 ms; v2 (chunked, divergence-free)
5.2 ms: ncu showed 255 registers/thread and 12.4 GB local-memory loads + 9.4 GB stores per launch (all families'
temporaries live in one kernel) at 16 % occupancy; v3 (one kernel per family, `wp.static` branches) 0.8 ms; v4
(transposed operands, coalesced loads/atomics) 0.25 ms.

E1 is **not** applied to the XLA path: there it measured 9 % slower at 3.9k (fusion side-effect), so the gather is kept
(commented in `moments.geometric_moments_packed`).

---

## D. Tried and not kept (reasons)

| experiment | why not |
|---|---|
| Fused GM contraction kernels (term lists, 3 iterations, session 1) | exact but only parity with XLA's factorised einsums at the time. **Superseded by E15** (family-specialised kernels + transposed operands: 1.31 -> 0.36 ms at 17.9k) |
| `n_unroll` (fori_loop unroll) | no gain; longer compiles |
| Async host-dispatched k-step blocks | removes syncs but unrolled kernels are slower; net loss without profiler |
| Sampling once per chunk + WHILE/CONDITIONAL command buffers | XLA does not run the loop on-device; no gain |
| Reverse-pair cache across steps | +2-3 % (noise); adds loop state and an `idx` third-row convention |
| `fast_math` in Warp kernels | +1.8 % (noise), VJP error 2.5x larger |
| Fused pair displacement custom VJP (E5) | exact, but the pair-cotangent `add_any` (moments + NLH) just moved; no gain |
| Global `--xla_gpu_enable_triton_gemm=false` | -32 to -45 % |
| E9: GM contractions as factorised broadcast-multiply-sum (no dot_general) | exact (2.5e-16), and at default precision it removes the remaining TF32 force error; 2.7x faster standalone (17.9k: 549 -> 203 us fwd+bwd), but in the full MD graph 19 % / 9 % SLOWER even with optimization barriers: XLA pushed the readout dot through the concatenated slices (138 rank-1 loop fusions/step) and built recomputing backward fusions (~0.5 ms each). Needs a hand-written kernel, not XLA. |
| E7: whole (2, P) pair array to Warp kernels instead of row slices | removes 4 x 10 MB copies/step, yet -5.6 % at 17.9k (reproducible, interleaved A/B); XLA kernels such as the contraction transpose (58 -> 118 us) and readout SiLU fusion (43 -> 88 us) slowed. Hypothesis: the copies act as graph barriers; without them independent graph branches run concurrently and contend with the Warp kernels. |
| Larger neighbour skin (1.0 / 1.5 A) | flat |
| E26: vector-packed radial coefficients in `gm_forward` (one length-4 vector load per basis function) | exact, but the kernel went 217 -> 469 us: Warp's generic-length vector accumulate does not compile to vector loads + FMAs |
| E17: per-family forward contraction kernels | kernels 105 -> 83 us, but zero-initialising the in/out feature array costs 27 us and 3.9k loses 1.8 % to extra launches |

## E. Known remaining costs (documented, not changed)
* Nose-Hoover chain update: a single-thread chain of fp64 scalar ops, ~23 us/step after E23 (was ~38), independent
  of N (`../perf_notes.md` N1, E23).
* jax-md neighbour update `cond`: copies `idx` (s32[2,P]) twice per step (XLA copy insertion around the conditional).
* XLA contraction backward: zero-broadcast accumulator copies f32[N,5,5,5] x4 and f32[N,5,5] x4 per step (gone with
  `contract_kernel: warp`).
* XLA schedule sensitivity at ~18k atoms (working set ~ L2 size): see the E14 note in C.
* Not performance, found while validating: `H5TrajHandler` (apax/md/io.py) builds `Atoms(numbers, positions,
  momenta=...)` without the simulation masses, so trajectory *velocities* are p / (default element mass); wrong for
  any custom-mass run (e.g. H mass 4). Dynamics are unaffected. One-line fix: pass `masses=`.
* Also found while validating: a 20k-step NVE run with `sampling_rate: 20` left only 1900 frames at 0.5 fs spacing
  (the first 0.95 ps) in md.h5; trajectory sampling / buffer flushing should be checked (`../perf_notes.md` V1).
* NLH repulsion pair loop (~210 us/step at 17.9k) and the pair-cotangent `add_any` it causes; fusing NLH into the Warp
  pair kernels is the lever.

## E2. Memory (peak GPU memory of the MD process, L40S 46 GB, current recommended config)
17.9k 1.04 GB, 143k 4.9 GB, 483k 9.3 GB, 858k 18.1 GB, 1.14M 34.7 GB (fits; before E15/E21 the 858k run needed
34.6 GB). The default jax-md builder needs far more for non-cubic boxes (19 GB at 71.5k, 34.6 GB at 322k): its
fractional cell list uses the shortest axis' cell count on all axes (`../perf_notes.md` M1).

## F. Portability (what is L40S-specific)

Measured only on an L40S (Ada, sm_89; fp64 = 1/64 fp32; 96 MB L2; 864 GB/s). Expected behaviour elsewhere:

| change | mechanism | expectation on other CUDA GPUs |
|---|---|---|
| A1 Warp moments, A2 pull, E2, E3a/b, E10, E11 | less work / no atomic contention / no pair-sized HBM intermediates | general; Warp JIT-compiles per architecture |
| B1 elementwise displacement | avoids Triton GEMM for 3x3 mat-vecs and TF32 | general on sm_80+ (TF32 exists from Ampere) |
| fp32 neighbour list, B2 no per-step inv(box) | avoids fp64 | largest gain on fp64-weak GPUs (Ada/Blackwell consumer, L40S); smaller on A100/H100 (fp64 1/2 rate) |
| E8 fused NLH, A4 capacity slack | fewer pair-sized memory passes | depends on L2 vs pair-array size: A100 (40 MB) / H100 (50 MB) spill at smaller N (fusion helps earlier) but have 2-3.4 TB/s |
| A3 highest precision | accuracy | cost depends on the fp32 : TF32 ratio |

Launch-parameter sensitivity: before E13, forward `n_split` 1-8 and block size 64-512 were within ~1.5 % (S1). After
E13-E21 the split matters at small N (3.9k: -2.5 % for the old default, -26 % for S = 32), so the heuristic was refitted
(E24) on three sizes. A runtime autotuner (time S in {1, 2, 4, 8} on the first neighbour list at MD setup, pass the
choice to the model) remains the robust option for other GPUs; not built. Tuned on L40S (re-check on other GPUs): forward `n_split` heuristic (target ~2^18 threads, gives 8), reduce split S = 4,
Warp block size 256, neighbour skin 0.5 A, and the E7 outcome (graph scheduling effect). Validation recipe:
`ecemc/warp_bench/bench.sh` on the original vs the recommended config, both box sizes (~2 min each).
