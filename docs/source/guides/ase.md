# Use a model in ASE

```python
from apax.md import ASECalculator
atoms.calc = ASECalculator("models/apax")      # model directory (contains best/ and config.yaml)
atoms.get_potential_energy(); atoms.get_forces()
```

## Constructor options

| Argument | Default | Meaning |
|---|---|---|
| `model_dir` | — | model directory, or a **list** of directories → ensemble (mean + uncertainty) |
| `calc_stress` | as trained (`model.calc_stress`, default off) | `True` to enable `get_stress()` |
| `calc_hessian` | `False` | `True` to enable `calc.get_hessian(atoms)` ({doc}`derivatives`) |
| `force_variance` | from config | per-atom force uncertainty for shallow ensembles |
| `transformations` | `[]` | biases/calibration ({doc}`uncertainty`) |
| `dr_threshold` | 0.5 Å | neighbour-list skin |
| `padding_factor` | 1.5 | neighbour-list headroom before reallocation |
| `disable_cell_list` | `False` | true for non-periodic systems |

Results after a call: `energy`, `forces`, `stress` (if enabled), plus `energy_uncertainty`, `forces_uncertainty`,
`energy_ensemble`, `forces_ensemble` for ensembles, and property heads by name. Units: eV, eV/Å, eV/Å³.

## Recipes

```python
from ase.optimize import BFGS
BFGS(atoms).run(fmax=0.01)                                   # relax positions

from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Langevin(atoms, 1 * units.fs, temperature_K=300, friction=0.01).run(1000)
```

Cell relaxation: `BFGS(FrechetCellFilter(atoms)).run(fmax=0.01)` with `ASECalculator(..., calc_stress=True)`;
stress is only correct for cells ≥ 2·`r_max` in every direction ({doc}`../reference/known_issues`).

## Many structures at once

```python
pred = calc.batch_eval(list_of_atoms, batch_size=64)   # copies with results attached (positions in Å)
```

## Performance notes

- The first call compiles (slow); reuse one calculator per system size.
- Small periodic cells (height < 2·`r_max`) automatically use a different neighbour-list backend.
- ASE-driven MD runs Python each step; for long or large MD use `apax md` ({doc}`md`), typically 10–100× faster.
