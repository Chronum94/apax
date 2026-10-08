# Prepare training data

## Formats

| File | Read with |
|---|---|
| `.h5`, `.h5md`, `.hdf5` | [znh5md](https://github.com/zincware/znh5md) (fast, recommended for large sets) |
| anything else (`.extxyz`, `.traj`, `vasprun.xml`, `OUTCAR`, …) | `ase.io.read(path, index=":")` |

Labels are taken from each structure's attached calculator results (`atoms.calc.results`). `ase.io.write` of
structures that carry results (e.g. read from DFT output, or with a `SinglePointCalculator`) stores them:

```python
from ase.calculators.singlepoint import SinglePointCalculator
atoms.calc = SinglePointCalculator(atoms, energy=E, forces=F)   # optionally stress=S
```

## From VASP (or other DFT codes)

```python
from glob import glob
from ase.io import read, write

frames = []
for f in sorted(glob("runs/*/vasprun.xml")):
    frames += read(f, index=":")          # every ionic step; use index="-1" for final structures only
write("train.extxyz", frames)
```

- apax reads `energy`. With Fermi smearing VASP also reports `free_energy`; ASE stores the energy extrapolated to zero
  smearing as `energy` (consistent with forces only for small smearing — use small `SIGMA`).
- Combine all calculations in one file (or train/val/test files). Use the same DFT settings throughout.

## Labels and units

| Label | Shape | Notes |
|---|---|---|
| `energy` | scalar | total energy of the structure |
| `forces` | (N, 3) | |
| `stress` | ASE convention (`atoms.get_stress()`, eV/Å³) | periodic data only; see {doc}`../reference/known_issues` before training on stress |
| `hessian` | (N, 3, N, 3) | loaded only if a `hessian` loss/metric exists; **not unit-converted** (eV/Å²) |
| custom properties | any | loaded only if named in `loss`/`metrics` ({doc}`physics_terms`); not unit-converted |

- Every structure must carry the same labels: a label missing in some frames is dropped for all (and a missing
  `stress` crashes training with `KeyError: 'stress'`).
- `data.energy_unit` (`eV` default, `kcal/mol`, `Hartree`, `kJ/mol`) and `data.pos_unit` (`Ang` default, `Bohr`)
  convert energies, forces and stresses on load; predictions are always eV, eV/Å, eV/Å³. Keep periodic data in Å.
- One dataset is either all periodic or all non-periodic; 2D-periodic cells are not supported.
- Atoms farther than `r_max` from all others feel no interaction (fine for genuine isolated atoms, wrong otherwise).

## Splits

| Setup | Config |
|---|---|
| one file, random split | `data_path: all.extxyz`, `n_train: 900`, `n_valid: 100` (sum ≤ number of frames) |
| your own split | `train_data_path: train.extxyz`, `val_data_path: val.extxyz` (`n_train`/`n_valid` ignored) |
| test set for `apax eval` | `test_data_path: test.extxyz` (recommended; see {doc}`evaluation`) |

The random split is reproducible (`seed`, default 1) and saved in `<model dir>/train_val_idxs.npz`.

## Reference energies (the shift)

apax first fits per-element energies so the network only learns the remainder:

```yaml
data:
  shift_method: per_element_regression_shift   # default
  shift_options: {energy_regularisation: 1.0}  # ridge strength
```

- **Large absolute energies** (all-electron codes, many elements, |E| ≫ 1000 eV): the default ridge of 1.0 biases the
  fit badly (errors of keV). Use `shift_options: {energy_regularisation: 1.0e-6}`, or isolated-atom energies:
  ```yaml
  shift_method: isolated_atom_energy_shift
  shift_options: {E0s: {1: -13.6, 8: -2041.3}}   # atomic number: energy in **eV** (not converted)
  ```
- `mean_atom_energy_shift`: one mean per-atom energy for all elements.

Scaling (`scale_method`): `per_element_force_rms_scale` (default), `mean_energy_rms_scale`,
`global_custom_scale` (`scale_options: {factor: …}`), `per_element_custom_scale` (`{factors: {Z: f}}`).

## Very different system sizes

The default dataset pads every structure to the largest one. For mixed sizes (e.g. 5 to 500 atoms) pad per batch:

```yaml
data:
  dataset:
    processing: pbp        # per-batch padding, multi-worker loading
    num_workers: 10
    atom_padding: 10       # pad atoms to a multiple of this
    nl_padding: 2000       # pad neighbour pairs to a multiple of this
```

## Checklist

- [ ] every frame has `energy` and `forces` (same labels in all frames)
- [ ] `energy_unit` / `pos_unit` match the file
- [ ] a held-out test file exists
- [ ] large absolute energies → `energy_regularisation: 1.0e-6` or `E0s`
