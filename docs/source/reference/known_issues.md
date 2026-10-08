# Known issues (apax 0.13)

Behaviour verified in apax 0.13.0 that differs from what you would expect. Each entry gives the workaround.

## Stress and NPT

:::{danger}
**Stress is wrong in two cases in 0.13.0** (checked against finite differences of the model energy):
- **Training on stress:** the stress used during training is wrong for every periodic cell → do not train on stress.
- **Inference on thin cells:** `ASECalculator(calc_stress=True)` and `apax md` NPT give wrong stress for cells with any
  height < 2·`r_max`. For cells ≥ 2·`r_max` in every direction, inference stress is correct (to ~1e-5 eV/Å³).

Energies and forces are unaffected.
:::

## Training

| Issue | Workaround |
|---|---|
| `apax template train --full` writes `transfer_learning: {base_model_checkpoint: null}`; training then crashes with `TypeError: argument should be a str or an os.PathLike object` (the config still validates) | delete the `transfer_learning` section unless fine-tuning |
| `data.batch_size` defaults to 1 (slow) | always set it |
| `latest/` checkpoint is written only every `ckpt_interval` (500) epochs; an interrupted shorter run restarts from epoch 0 | set `ckpt_interval` to e.g. 10 |
| a label missing from some frames is dropped for all; if that label is in `loss`/`metrics` (e.g. stress) training fails with a bare `KeyError` | make every frame carry the same labels |
| `n_train + n_valid` larger than the dataset → `IndexError` | reduce them |
| `loss_type` typos (e.g. `msee`) pass `apax validate` and fail at training | check against {doc}`train_config` |
| `model.name: so3krates` needs the `myrto` package, which no extra installs → crash at training start | use `gmnn` or `equiv-mp` |

## Evaluation

| Issue | Workaround |
|---|---|
| without `test_data_path`, the test set includes the validation structures | always provide `test_data_path` |
| with `test_data_path`, the last structure is skipped | add a dummy last structure, or use `ASECalculator.batch_eval` |
| `test_idxs.npz` is only written without `test_data_path`; `eval.log` goes to the working directory | — |

## MD (`apax md`)

| Issue | Workaround |
|---|---|
| a box narrower than 2·`r_max` only logs `cutoff larger than half the box` and runs on with **wrong energies** (also mid-run in NPT); `disable_cell_list` does not help | replicate the cell so every height ≥ 2·`r_max` |
| neighbour-list reallocation (density increase, NPT) can crash with a shape `TypeError` | set `extra_capacity` > 0 (raise it until no reallocation is logged in `md.log`) and a larger `dr_threshold` |
| `stress` stored in `md.h5` is stress × volume (eV), not eV/Å³ | divide by the cell volume |
| `time`/`step` in `md.h5` are frame index × `dt` (off by `sampling_rate`) | multiply by `sampling_rate` |
| killed before the first trajectory flush (`buffer_size` frames) → no `md.h5`, restart crashes | keep `buffer_size` moderate (100–1000) |
| a killed-and-restarted run can contain duplicated frames, and its `step`/`time` arrays end up shorter than the data | prefer extending finished runs (raise `duration`) over killing; check frame counts after a restart |
| after a dynamics check stops a run, re-running restarts from the checkpoint before the stop and stops again (exit code 0) | inspect `md.log`; change the setup (or delete `ckpts/`) before resubmitting |
| the compiled MD loop is not stored in the JAX compilation cache (it contains a host callback) | every job recompiles; amortise with long jobs |
| dynamics checks are evaluated on saved frames only and stop at the end of the current `n_inner` block; frames beyond the threshold are already written; exit code 0 | lower `n_inner`/`sampling_rate` for tight control; check `md.log` |
| NVT ignores `thermostat_chain` settings | — |
| `reflection` check assumes an orthorhombic, periodic cell | — |
| performance summary includes compile time and is wrong after restarts | time long runs yourself |

## Uncertainty and selection

| Issue | Workaround |
|---|---|
| calibration factors only apply through `ASECalculator(transformations=...)`; `apax md` dynamics checks use raw uncertainties | scale MD thresholds by the inverse calibration factor |
| `kernel_selection` with the `identity` feature map raises `TypeError` | use `ll_grad` |
