# Descriptor Workshop — Improvement Roadmap

Baseline: `GaussianMomentDescriptor` + `RadialFunction` + `AtomisticReadout`.
Default config: `n_radial=5`, `n_basis=7`, `n_contr=8`, `n_species=119`.

---

## Parameter Budget (baseline)

| Component | Parameters | Notes |
|---|---|---|
| Species-pair embedding `[119, 119, 5, 7]` | 496,685 | ~99% unused for typical systems |
| Basis function means/widths | 0 | fully fixed |
| MLP `360→256→256→1` | ~158k | |
| `PerElementScaleShift` | ~238 | |
| **Total** | **~655k** | |

Descriptor dimension breakdown for `n_radial=5`:

| Contraction | Dim | Scaling |
|---|---|---|
| `contr_0` (M0) | 5 | n_radial |
| `contr_1` (M1·M1) | 15 | tril_2d(n_radial) |
| `contr_2` (M2:M2) | 15 | tril_2d(n_radial) |
| `contr_3` (M3:M3) | 15 | tril_2d(n_radial) |
| `contr_4` (M2·M2·M2) | 35 | tril_3d(n_radial) |
| `contr_5` (M1⊗M1:M2) | 75 | tril_2d × n_radial |
| `contr_6` (M3:M3:M2) | 75 | tril_2d × n_radial |
| `contr_7` (M3:M2⊗M1) | 125 | n_radial³ |
| **Total** | **360** | |

`contr_5/6/7` scale as O(n_radial³) — increasing `n_radial` is expensive.

---

## 1. Radial Basis Functions (`basis_functions.py: GaussianBasis`)

### 1a. Non-uniform basis spacing
**Problem:** Linear spacing from `r_min` to `r_max` wastes resolution. The repulsion wall
and bonded region (`~0.8–2.5 Å`) need far more basis functions than the diffuse long-range
region (`~4–6 Å`).

**Options (increasing flexibility):**
- Log/exponential spacing: `shifts = r_min + (r_max - r_min) * (exp(t_i) - 1) / (e - 1)`
- Learnable means via `softplus`-transformed parameters + cumsum to enforce ordering
- Learnable per-basis widths (separate `β_k` per Gaussian instead of one global `β`)

**Note:** `BesselBasis` (current default in `GMNNConfig`) does not have this problem since
its functions are not localised around fixed centres. But Gaussian basis + log spacing is
still worth exploring for fine-grained short-range resolution.

### 1b. C∞ cutoff envelope
**Problem:** The cosine cutoff `0.5*(cos(π·r/r_max)+1)` is C¹ at `r_max`. The second
derivative is discontinuous there, producing kinks in the PES that corrupt phonon
frequencies and elastic constants (stress is a second-order perturbation).

**`EquivMPRepresentation`** already uses `e3x.nn.smooth_cutoff` (C∞). The GMNN
descriptor should use the same envelope.

A practical C∞ option:
```python
# p=5 → C², higher p → smoother
envelope = ((1 - (r / r_max) ** p) ** 2) * (r < r_max)
```

**Effort:** Low. Zero new dependencies (e3x is already present).

### 1c. Enriched distance features
Instead of raw `r` as basis input, consider:
- `log(r)`: equal resolution per decade, better for wide distance ranges
- `[1/r, 1/r⁶, 1/r¹²]`: physically motivated for electrostatics / dispersion; natural
  complement to the `LatentEwald` correction already in the codebase

---

## 2. Element Embeddings (`basis_functions.py: RadialFunction`)

### 2a. Factored element embeddings  ← highest ROI change
**Problem:** The current embedding table has shape `(n_species, n_species, n_radial,
n_basis)`. For `n_species=119` this is 496k parameters, of which ~99% are never updated
for typical systems (10–20 elements). Parameter count scales as O(n_species²).

**Proposed: per-element embedding + shared projection**
```
e_Z ∈ ℝ^d              per-element lookup table  (119 × d params)
pair_feat = e_Zi ⊕ e_Zj   or   e_Zi ⊙ e_Zj      (symmetric options below)
W: ℝ^{2d} → ℝ^{n_radial × n_basis}               one shared linear layer
radial_weights = W(pair_feat)                      (n_pairs, n_radial, n_basis)
```

Parameter count for `d=16`:
- Table: `119 × 16 = 1,904`
- Projection: `32 × 35 = 1,120`
- **Total: ~3k vs. 496k — 165× reduction**

Symmetric pair combination options (automatically enforce U(A,B)=U(B,A)):
- Sum: `e_Zi + e_Zj`  → projection input is `d`-dim
- Hadamard: `e_Zi ⊙ e_Zj`  → projection input is `d`-dim
- Concat + symmetrize: `[e_Zi, e_Zj] + [e_Zj, e_Zi]`  → input is `2d`-dim, symmetric

### 2b. Symmetrize existing pair embeddings (zero-cost fix)
**Problem:** `self.embeddings[Z_j, Z_i]` and `self.embeddings[Z_i, Z_j]` are independent
parameters. The pair interaction U(A,B)=U(B,A) gives no physical reason for asymmetry.
The model wastes capacity fitting anti-symmetric components that should be zero.

**Fix (one line, current architecture):**
```python
species_pair_coeffs = 0.5 * (self.embeddings[Z_j, Z_i] + self.embeddings[Z_i, Z_j])
```

**Effort:** Trivial.

### 2c. Scalar message-passing refinement of element embeddings
**Problem:** Every `H` atom has the same embedding regardless of whether it's in C-H,
O-H, or N-H. Species context only enters per-pair through the radial weights, with no
cross-species communication.

**Proposed: one scalar MP step before descriptor computation**
```
e_Z^(0) = embed(Z)                                       d-dim learned lookup
e_Z^(1) = e_Z^(0) + MLP( Σ_j  f_r(r_ij) · e_Zj^(0) )  scalar gate, no angular feats
```
Use `e_Z^(1)` as element context in the radial function weights.

- Cost: O(n_pairs · d) — negligible
- Effect: one step doubles the effective range to 2·r_max; two steps → 3·r_max
- `EquivMPRepresentation` with `max_degree=0` is architecturally identical —
  the module already exists in the codebase

**Effort:** Medium.

### 2d. Physical priors on element embeddings
Initialize element embeddings from periodic-table properties (electronegativity, covalent
radius, valence, period, group, ionization energy) via a small learned projection:
```python
e_Z^(0) = W_phys @ phys_features[Z]   # W_phys: d × n_props, learnable
```
Provides a warm start that reduces data needed for convergence and prevents chemically
similar elements (O/S, F/Cl) from diverging arbitrarily.

**Effort:** Low.

---

## 3. Direction Normalization (`gaussian_moment_descriptor.py`)

### 3a. Safe norm (trivial correctness fix)
**Problem:**
```python
dr_repeated = einops.repeat(dr + 1e-5, "neighbors -> neighbors 1")
dn = dr_vec / dr_repeated
```
The `1e-5` offset biases unit vectors for **all** distances, not just near-zero ones.
The gradient `∂(dn)/∂(dr_vec)` is wrong everywhere because the denominator is `dr + 1e-5`
rather than `dr`.

**Fix:**
```python
safe_dr = jnp.where(dr < 1e-8, jnp.ones_like(dr), dr)
dn = jnp.where(dr[..., None] < 1e-8, jnp.zeros_like(dr_vec), dr_vec / safe_dr[..., None])
```
At `dr=0`, `dn=0` is physically correct (no directional information → all angular moments
are zero). No bias introduced at any other distance.

**Effort:** Trivial.

---

## 4. Geometric Moment Contractions (`gaussian_moment_descriptor.py`)

### 4a. Descriptor conditioning — LayerNorm before MLP
**Problem:** The 8 contractions have very different expected magnitudes:
- `contr_0`: O(N_neigh · φ) — linear in neighbour count
- `contr_1–3`: O(N_neigh · φ²) after einsum + segment_sum
- `contr_5–7`: 3-body, O(N_neigh³ · φ³) in principle

Without normalization the MLP input is poorly conditioned, particularly when `N_neigh`
varies across training configurations.

**Options:**
- 8 learnable per-contraction scale scalars (initialized to 1.0) before concatenation
- `LayerNorm` applied to the full 360-dim descriptor vector before MLP input
- Divide `contr_k` by `N_neigh^k` for expected-value stability across system sizes

**Effort:** Low.

### 4b. Additional invariant contractions
The 8 contractions are not a complete basis for 3-body rotational invariants. A known
missing one:
```python
# M1 · M2 · M1 — captures asymmetric angular environments
contr_extra = jnp.einsum("ari, arsj, atj -> arst", moments[1], moments[2], moments[1])
```
Also: simple products of existing scalar contractions (e.g., `contr_0 * contr_1`) are valid
invariants not currently included.

**Effort:** Medium.

### 4c. Verify tril_3d indexing symmetry
`tril_3d_indices` includes `[i, j, j]` for **all** `j ≠ i` (both `j < i` and `j > i`).
This differs from the standard lower-triangular set, which would only include `j ≥ i`.
Whether this matches the symmetry of the `contr_4` einsum
`"arij, asik, atjk -> arst"` should be verified — some of these index triples may be
redundant, wasting descriptor dimensions.

**Effort:** Low (analysis) / Medium (fix + retest).

---

## 5. Flexible Radial Function via MLP

**Problem:** The current `einsum(species_pair_embedding, basis)` is a bilinear map —
it can only represent linear combinations of fixed basis functions with species-dependent
coefficients. It cannot learn non-linear, species-conditioned radial shapes.

**Proposed: MLP radial function**
```
basis(r) → ℝ^n_basis          fixed encoding (keep Bessel/Gaussian)
e_Zi, e_Zj → ℝ^d              per-element embeddings (Section 2a)

φ_ij(r) = MLP( concat(basis(r), e_Zi, e_Zj) ) · envelope(r)
           [n_basis + 2d] → [32] → [n_radial]
```
This is strictly more expressive than the current bilinear form, can represent arbitrary
non-linear species-conditioned radial functions, and costs one small MLP per neighbor pair
(JIT-compilable, vmap-able). Used in PaiNN, SCFNet, and variants.

**Effort:** Medium.

---

## 6. Descriptor Compression Layer

**Problem:** The 360-dim descriptor is passed directly to the first MLP layer (size 256),
giving a `360 × 256 = 92k` weight matrix as the dominant parameter block.

**Proposed: linear bottleneck before MLP**
```
G (360) → Linear(360 → d_comp) → MLP(d_comp → 256 → 1)
```
- For `d_comp=64`: first block shrinks from 92k to `360×64 + 64×256 = 39k` (−57%)
- Can be initialized from top-`d_comp` PCA components of training set descriptors
- Allows the model to learn a compressed, task-relevant representation of the descriptor

This is especially useful when `n_radial` is increased (descriptor grows as O(n_radial³)).

**Effort:** Low.

---

## 7. Summary: Impact vs. Effort

| # | Change | Model size | Accuracy | Stability | Speed | Effort |
|---|---|---|---|---|---|---|
| 2a | Factored element embeddings | ✅✅✅ | ✅ | ✅ | ✅ | Medium |
| 2b | Symmetrized pair embeddings | ✅ | ✅ | — | — | **Trivial** |
| 3a | Safe norm for direction | — | ✅ | ✅ | — | **Trivial** |
| 1b | C∞ cutoff envelope | — | ✅✅ | ✅ | — | Low |
| 4a | Descriptor LayerNorm | — | ✅ | ✅✅ | — | Low |
| 2c | Scalar MP element embeddings | ✅ | ✅✅✅ | — | ⚠️ | Medium |
| 5 | MLP radial function | — | ✅✅ | — | ⚠️ | Medium |
| 6 | Descriptor compression layer | ✅ | — | — | ✅✅ | Low |
| 4b | Additional contractions | — | ✅ | — | ⚠️ | Medium |
| 2d | Physical priors on embeddings | — | ✅ | ✅ | — | Low |
| 1a | Learnable basis positions/widths | — | ✅ | — | — | Medium |
| 4c | Verify/fix tril_3d indexing | — | ✅ | — | — | Low |

⚠️ = slight inference cost increase

---

## Recommended Implementation Order

1. **Trivial wins** (2b, 3a): symmetrize pair embeddings, safe norm. Zero risk, zero cost.
2. **Low-effort wins** (1b, 4a): C∞ envelope, descriptor LayerNorm.
3. **Core refactor** (2a): factored element embeddings. Replaces the 119×119 table.
   Prerequisite for 2c and 5.
4. **Accuracy push** (2c): scalar MP refinement of element embeddings.
5. **Flexibility** (5): MLP radial function, conditioned on element embeddings.
6. **Completeness** (4b, 1a): additional contractions, learnable basis.
