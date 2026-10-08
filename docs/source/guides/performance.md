# Performance and memory

## Training speed

| Lever | Effect |
|---|---|
| GPU (`apax[cuda]`) | 10–100× over CPU for realistic models |
| `data.batch_size` | default 1 is slow; 8–32 for small systems |
| `model.nn` | `[256, 256]` → `[128, 128]` or `[64, 64]`: faster, often equally accurate on small data |
| `model.basis.r_max` | pairs (and cost) grow ~r³: 6 → 5 Å ≈ 1.7× fewer pairs |
| `data.dataset.processing: pbp` | avoids padding everything to the largest structure (mixed sizes) |
| more GPUs | used automatically (`data_parallel: true`) |
| shallow ensemble with `force_variance` | ≈ 4–5× slower than a single model ({doc}`uncertainty`) |

## MD speed

Knobs in {doc}`md`: larger `n_inner`, a suitable `dr_threshold`, drop `stress` from `properties` when not needed,
`disable_cell_list: true` for non-periodic systems. `md.log` reports ns/day and µs/step/atom (includes compile
time). `export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax` caches compiled training and ASE kernels across jobs;
the `apax md` loop recompiles at every start.

## GPU memory

apax sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` on import, so JAX allocates on demand. Out-of-memory fixes:
lower `batch_size` / `valid_batch_size`; lower `r_max`; for very mixed sizes use `processing: pbp`.
To cap usage set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` in the environment.

## Compilation

The first epoch/step compiles (seconds to minutes). New input shapes trigger recompiles: in MD, neighbour-list growth
(set `extra_capacity`); in ASE, a different number of atoms. Reuse one `ASECalculator` per system size.

## Precision

Descriptor and readout run in fp32, the per-element scale/shift in fp64 (`descriptor_dtype`, `readout_dtype`,
`scale_shift_dtype`). fp32 is accurate for MD; use fp64 for finite-difference checks or very tight relaxations.
