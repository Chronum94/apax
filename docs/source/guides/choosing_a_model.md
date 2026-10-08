# Choosing a model (and coming from MACE)

## Models

| `model.name` | Type | Receptive field | Use when | Key hyperparameters |
|---|---|---|---|---|
| `gmnn` (default) | Gaussian moments, invariant, strictly local | `r_max` | default choice: fast training and MD, good accuracy per cost | `basis.r_max`, `n_radial` (3–5), `nn`, `n_contr` |
| `equiv-mp` | equivariant message passing | `r_max` × `num_iterations` (local for the default 1) | equivariant features; slower | `features`, `max_degree`, `num_iterations` |
| `so3krates` | equivariant attention | grows with `num_layers` | research; requires the extra `myrto` package (not installed by any extra) | `num_layers`, `num_features`, `max_degree`, `num_heads` |

All share `basis`, `nn`, `ensemble`, `empirical_corrections`, `property_heads`, `calc_stress`, `calc_hessian`
({doc}`../reference/train_config`). Cost scales with the number of neighbour pairs (~`r_max`³) and readout width.

## MACE → apax key translation

| MACE | apax |
|---|---|
| `--r_max` | `model.basis.r_max` |
| `--batch_size` / `--valid_batch_size` | `data.batch_size` / `data.valid_batch_size` |
| `--train_file`, `--valid_file`, `--test_file` | `data.train_data_path`, `val_data_path`, `test_data_path` |
| `--valid_fraction` | `data.data_path` + `n_train`/`n_valid` |
| `--E0s` (dict) | `data.shift_method: isolated_atom_energy_shift`, `shift_options: {E0s: {Z: E}}` (eV) |
| `--E0s average` | `data.shift_method: per_element_regression_shift` (ridge via `energy_regularisation`) |
| `--energy_weight` (weighted loss: per-atom energy) | energy `weight` with `atoms_exponent: 2` |
| `--forces_weight` | forces `weight` = MACE value / 3 (apax sums over x, y, z) |
| `--stress_weight` | stress loss is on stress × volume (eV): not directly comparable |
| `--loss huber` | `loss_type: huber`, `parameters: {delta: ...}` |
| `--ema --ema_decay` | `weight_average` (updated twice per epoch, not per step; no direct equivalent) |
| `--lr`, `--scheduler` | `optimizer.nn_lr`/`emb_lr`/…, `optimizer.schedule` |
| `--patience` | `patience` |
| `--max_num_epochs` | `n_epochs` |
| `--foundation_model` / fine-tuning | `transfer_learning` ({doc}`fine_tuning`) |
| committee of models | `model.ensemble` or `ASECalculator([dir1, dir2, ...])` |
| `--default_dtype float64` | `descriptor_dtype`, `readout_dtype` (`fp32` default) |
| `--device cuda` | automatic (GPU if `apax[cuda]` is installed) |
| `--num_channels`, `--max_L`, `--num_interactions` | `equiv-mp`: `features`, `max_degree`, `num_iterations` (GMNN: `n_radial`, `n_contr`, `nn`) |
| `--correlation` | no equivalent |

Not available: SWA/stage-two loss switching, per-config weights, `ReduceLROnPlateau`.
