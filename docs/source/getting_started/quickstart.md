# Quickstart (≈5 minutes, CPU)

Train a potential for copper on toy data, test it, use it in ASE and run MD. Every command below is copy-pasteable
and was tested on a laptop CPU (training takes 1.5–3 min there; seconds on a GPU).

## 1. Make a dataset

Any file ASE can read works if every structure carries energies and forces (stress optional).
Here we label rattled copper cells with ASE's cheap EMT potential instead of DFT:

```python
# make_data.py
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

rng = np.random.default_rng(0)
frames = []
for i in range(300):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)          # 32 atoms
    atoms.rattle(stdev=0.1, seed=i)
    atoms.set_cell(atoms.cell * rng.uniform(0.97, 1.03), scale_atoms=True)
    atoms.calc = EMT()
    e, f, s = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
    atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f, stress=s)  # store labels
    frames.append(atoms)
write("cu_train.extxyz", frames[:250])   # training + validation
write("cu_test.extxyz", frames[250:])    # held-out test set
```

```bash
python make_data.py
```

## 2. Write a training config

```bash
apax template train        # creates config.yaml
```

Replace the `<...>` placeholders, add the `test_data_path` line and adjust the data sizes so that `config.yaml` reads:

```yaml
n_epochs: 50

data:
  directory: models
  experiment: apax          # model is saved to models/apax/
  data_path: cu_train.extxyz
  test_data_path: cu_test.extxyz
  n_train: 225
  n_valid: 25               # validation set picks the best checkpoint
  batch_size: 4
  valid_batch_size: 25

metrics:
  - name: energy
    reductions: [mae]
  - name: forces
    reductions: [mae, mse]

loss:
  - name: energy
  - name: forces
    weight: 1.0
```

```bash
apax validate train config.yaml     # catches typos before training
```

## 3. Train (1.5–3 min on CPU)

```bash
apax train config.yaml
```

The first epoch is slow (JAX compiles the model); later epochs take about 2 s here. Repeated
`Waiting for previous save to complete` warnings are harmless.
Output in `models/apax/`: `best/` (checkpoint with lowest validation loss), `log.csv` (per-epoch metrics), `config.yaml`
(the fully expanded config with all defaults). Expect validation errors of roughly 0.02 eV per structure
(< 1 meV/atom for these 32-atom cells) and 0.03 eV/Å for forces.

## 4. Measure test error

```bash
apax eval config.yaml
tail -1 models/apax/eval/log.csv    # last row: averages over the test set (currently skips its last structure)
```

## 5. Use the model in ASE

```python
# use_model.py  (don't name scripts ase.py or apax.py: they shadow the packages)
from ase import units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from apax.md import ASECalculator

atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(3)
atoms.calc = ASECalculator("models/apax")          # the experiment directory
print(atoms.get_potential_energy(), atoms.get_forces().shape)     # eV, (108, 3) eV/Å

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Langevin(atoms, 2 * units.fs, temperature_K=300, friction=0.01).run(100)
```

Anything that takes an ASE calculator works: relaxations, NEB, phonons, ASE MD.

## 6. Run fast MD with JAX-MD

```bash
python -c "from ase.build import bulk; from ase.io import write; write('start.extxyz', bulk('Cu','fcc',a=3.6,cubic=True).repeat(3))"
apax template md            # creates md_config.yaml
```

In `md_config.yaml` set `T0: 300`, `duration: 2000` (fs) and `initial_structure: start.extxyz`, then:

```bash
apax md config.yaml md_config.yaml      # 2 ps NVT, ~20 s on CPU
```

The trajectory is `md/md.h5` (H5MD); read it with `znh5md`:

```python
import znh5md
traj = znh5md.IO("md/md.h5")   # list-like of ase.Atoms with energies/forces
print(len(traj), traj[-1].get_potential_energy())
```

## Next

- Your own data (VASP, other codes): {doc}`../guides/data`.
- Is the model good enough? {doc}`../guides/evaluation`.
- Better models: {doc}`../guides/training`.
- Everything a config can contain: {doc}`../reference/train_config`.
