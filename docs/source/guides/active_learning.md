# Active learning: pick new structures to label

Pick the most informative structures from a candidate pool (e.g. MD snapshots) so the labelling budget goes where
the model is weakest.

## Select from a pool

```python
from ase.io import read
from apax.bal.api import kernel_selection

train = read("train.extxyz", ":")
pool = read("md_snapshots.extxyz", ":")

ranking, distances, g_train, g_pool = kernel_selection(
    "models/apax",              # or a list of model dirs (ensemble)
    train,
    pool,
    base_fm_options={"name": "ll_grad", "layer_name": "dense_2"},
    selection_method="max_dist",  # greedy farthest-point selection
    selection_batch_size=20,      # how many to pick
)
to_label = [pool[i] for i in ranking]
```

| Feature map (`base_fm_options.name`) | Options | Notes |
|---|---|---|
| `ll_grad` | `layer_name` | gradient of the energy w.r.t. the last readout layer (default choice) |
| `ll_force_feat` | `layer_name`, `strategy: flatten` | force-based features |
| `full_grad_rp` | `num_rp: 512` | random projection of all-parameter gradients |

`layer_name` is the last dense layer, `dense_<len(model.nn)>`: `dense_2` for the default two hidden layers,
`dense_3` for `nn: [64, 32, 16]`.

## A complete loop

Train an ensemble → sample with MD (stop when uncertain) → select → label → add → retrain. Runnable as is with the
toy EMT labels from the {doc}`../getting_started/quickstart` (copy `cu_train.extxyz` to `train.extxyz`);
replace `label()` with your DFT call. About 7 min on a laptop CPU.

```python
# al_loop.py - active learning: train ensemble -> MD -> select -> label -> retrain
import subprocess

import numpy as np
import yaml
import znh5md
from ase.build import bulk
from ase.io import read, write
from apax.bal.api import kernel_selection
from apax.md import ASECalculator


def label(atoms):  # replace with your DFT calculation
    from ase.calculators.emt import EMT
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = atoms.copy()
    atoms.calc = EMT()
    e, f = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f)
    return atoms


train_cfg = {
    "n_epochs": 30,
    "data": {
        "directory": "models", "experiment": "iter0", "data_path": "train.extxyz",
        "n_train": 225, "n_valid": 25, "batch_size": 8, "valid_batch_size": 25,
    },
    "model": {"name": "gmnn", "nn": [64, 64],
              "ensemble": {"kind": "shallow", "n_members": 4, "force_variance": True}},
    "loss": [{"name": "energy"}, {"name": "forces", "weight": 4.0}],
    "metrics": [{"name": "energy", "reductions": ["mae"]},
                {"name": "forces", "reductions": ["mae"]}],
}
write("start.extxyz", bulk("Cu", "fcc", a=3.6, cubic=True).repeat(3))

for it in range(2):
    model_dir = f"models/iter{it}"
    train_cfg["data"]["experiment"] = f"iter{it}"
    yaml.safe_dump(train_cfg, open("config.yaml", "w"))
    subprocess.run(["apax", "train", "config.yaml"], check=True)            # 1. train

    # threshold = 4 x the 95th percentile of the force uncertainty on validation frames
    train = read("train.extxyz", ":")
    val = [train[i] for i in np.load(f"{model_dir}/train_val_idxs.npz")["val_idxs"]]
    calc = ASECalculator(model_dir)
    unc = []
    for atoms in val:
        atoms = atoms.copy()
        atoms.calc = calc
        atoms.get_potential_energy()
        unc.append(calc.results["forces_uncertainty"].max())
    threshold = 4 * float(np.percentile(unc, 95))

    md_cfg = {
        "ensemble": {"name": "nvt", "dt": 1.0,
                     "temperature_schedule": {"name": "constant", "T0": 1500}},
        "duration": 2000, "n_inner": 100, "sim_dir": f"md{it}",
        "initial_structure": "start.extxyz",
        "properties": ["energy", "forces", "forces_uncertainty"],
        "dynamics_checks": [{"name": "forces_uncertainty", "threshold": threshold}],
    }
    yaml.safe_dump(md_cfg, open("md_config.yaml", "w"))
    subprocess.run(["apax", "md", "config.yaml", "md_config.yaml"], check=True)   # 2. sample

    pool = znh5md.IO(f"md{it}/md.h5")[:]                                    # 3. select
    idx, *_ = kernel_selection(
        model_dir, train, pool,
        base_fm_options={"name": "ll_grad", "layer_name": "dense_2"},
        selection_method="max_dist", selection_batch_size=10,
    )
    new = [label(pool[i]) for i in idx]                                     # 4. label
    write("train.extxyz", train + new)                                      # 5. grow data
    train_cfg["data"]["n_train"] += len(new)
    print(f"iteration {it}: threshold {threshold:.2f} eV/Å, max seen "
          f"{max(a.calc.results['forces_uncertainty'].max() for a in pool):.2f}, added {len(new)}")
# retrain once more on the final train.extxyz before production use
```

Notes:
- Run `apax md` as a separate process per iteration (calling `run_md` twice in one Python process currently fails).
- The threshold is derived from the validation set each iteration. If the check never fires, the whole trajectory
  is the selection pool (that is fine); raise the temperature or lower the factor to explore harder.
- Retrain once more after the last iteration: the final selected structures are not yet in a model.
- Each iteration here retrains from scratch, the safest choice. To save time, fine-tune the previous model instead
  ({doc}`fine_tuning`) — at the same level of theory do **not** reset `scale_shift`.
- Stop when MD at the target conditions no longer triggers the checks and DFT re-labelled snapshots match test errors
  ({doc}`evaluation`).
