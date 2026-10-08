# Troubleshooting

Also see {doc}`../reference/known_issues` for confirmed bugs and their workarounds.

## Config and startup

| Symptom | Cause → fix |
|---|---|
| `Unable to extract tag using discriminator 'name'` under `model` | a `model:` section needs `name: gmnn` (or `equiv-mp`, `so3krates`) |
| `Extra inputs are not permitted` / `Field required` | misspelt or missing key. `apax validate train config.yaml`; list keys with `apax schema train model.gmnn --keywords` (any dotted section) or `apax schema train --flat` |
| `TypeError: argument should be a str or an os.PathLike object` at training start | `--full` template's `transfer_learning: {base_model_checkpoint: null}` → delete the section |
| `IndexError` while splitting data | `n_train + n_valid` exceeds the number of frames |
| `config.yaml already exists` from `apax template` | the template command never overwrites; delete or rename the file |
| `so3krates` crashes at start (`NoneType` not callable) | needs the separate `myrto` package |

## Training

| Symptom | Cause → fix |
|---|---|
| very slow first epoch | JAX compilation; normal |
| repeated `WARNING ... Waiting for previous save to complete took ... seconds` | checkpoint writing (orbax) is slower than an epoch; harmless — raise `ckpt_interval` to silence |
| `KeyError: 'forces_uncertainty'` with a force `nll` loss | set `force_variance: true` in the shallow ensemble |
| training slow overall | `batch_size` defaults to 1: set 8–32; check `python -c "import jax; print(jax.devices())"` shows a GPU |
| energy errors of keV / huge initial loss | large absolute energies with the default shift ridge → `shift_options: {energy_regularisation: 1.0e-6}` or E0s ({doc}`data`) |
| errors wrong by a constant factor | wrong `energy_unit` / `pos_unit` |
| `KeyError: 'stress'` at the first step | stress is in `loss`/`metrics` but some frames lack it: add it everywhere or remove the stress loss/metric |
| a label silently missing from `log.csv` | it is absent in at least one frame (logged as an error at start) |
| resumed training starts from epoch 0 | `latest/` saved only every `ckpt_interval` (500) epochs: lower it |
| `loss_type` typo accepted by `validate` but crashes | check the value list in {doc}`../reference/train_config` |

## Inference and MD

| Symptom | Cause → fix |
|---|---|
| `get_stress` / `get_hessian` raises `PropertyNotImplementedError` | create the calculator with `calc_stress=True` / `calc_hessian=True` |
| energies differ between `apax md` and ASE for a small cell | cell height < 2·`r_max`: replicate the cell |
| MD explodes / NaN | timestep too large (0.5 fs; 0.25 fs with H), no short-range repulsion ({doc}`physics_terms`), or the structure leaves the training distribution (`dynamics_checks`, {doc}`uncertainty`) |
| NaN abort and no `md.h5` | frames are written every `buffer_size` frames; lower it for exploratory runs |
| crash with a shape `TypeError` after the neighbour list reallocates | set `extra_capacity` and a larger `dr_threshold` |
| GPU not used | install `apax[cuda]` on Linux; check `jax.devices()` |

Still stuck: [Discord](https://discord.gg/7ncfwhsnm4) or [open an issue](https://github.com/apax-hub/apax/issues)
with your `config.yaml`, `apax --version` and the full error.
