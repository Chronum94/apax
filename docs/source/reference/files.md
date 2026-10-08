# Files apax reads and writes

## Model directory (`data.directory/data.experiment`)

| Path | Written | Content |
|---|---|---|
| `config.yaml` | start of training | the full config with every default filled in (what `ASECalculator` and `apax md` read) |
| `best/` | whenever `val_loss` improves | checkpoint used by all inference tools |
| `latest/` | every `ckpt_interval` epochs | checkpoint for resuming |
| `log.csv` | every epoch | `epoch`, `epoch_time`, `train_loss`, `val_loss`, `{train,val}_{metric}_{reduction}` |
| `train.log` | during training | text log |
| `train_val_idxs.npz` | start (with `data_path`) | `train_idxs`, `val_idxs` into the data file |
| TensorBoard events | with the `tensorboard` callback | |
| `eval/log.csv`, `eval/test_idxs.npz` | `apax eval` | running-average test metrics (last row = result), test indices |

To share a model, copy the whole directory (at least `config.yaml` and `best/`).

## MD (`sim_dir`)

| Path | Content |
|---|---|
| `md.h5` (`traj_name`) | H5MD trajectory: positions, velocities, `properties`; read with `znh5md.IO("md.h5")`. Stored stress is stress × volume (eV); `time`/`step` are frame index × `dt` ({doc}`known_issues`) |
| `md.log` | log with the performance summary |
| `ckpts/` | restart checkpoints |

## Input data

`.h5`/`.h5md`/`.hdf5` via znh5md, everything else via `ase.io.read`. Labels from `atoms.calc.results`. See {doc}`../guides/data`.
