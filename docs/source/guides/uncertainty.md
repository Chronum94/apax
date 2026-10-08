# Uncertainty and ensembles

## Train an ensemble

```yaml
model:
  name: gmnn
  ensemble:
    kind: shallow            # n readout heads sharing one descriptor
    n_members: 8
    force_variance: true     # per-atom force uncertainty (default: true)
```

| Option | Cost vs one model (8 members) | Notes |
|---|---|---|
| `shallow`, `force_variance: false` | small (estimate ≈ 1.1–1.5×) | energy uncertainty only; `nll`/`crps` on energy only |
| `shallow`, `force_variance: true` (default) | ≈ 4.5× (measured) | one backward pass per member; needed for force `nll` |
| `full` | n× | n independent models trained in parallel |
| several model dirs in `ASECalculator([...])` | n× | train models separately (e.g. different seeds) |

For calibrated spreads (not just correlated with error), train with a probabilistic loss:

```yaml
loss:
  - name: energy
    loss_type: nll       # or crps; shallow ensembles only
  - name: forces
    loss_type: nll
    weight: 4.0
```

## Read uncertainties

```python
from apax.md import ASECalculator
calc = ASECalculator("models/ens")
atoms.calc = calc
atoms.get_potential_energy()
calc.results["energy_uncertainty"]      # eV, std over members (per structure)
calc.results["forces_uncertainty"]      # (N, 3) eV/Å
calc.results["forces_ensemble"]         # per-member forces
```

## Calibrate on validation data

```python
from ase.io import read
from apax.calibration import compute_calibration_factors
from apax.md.function_transformations import GlobalCalibration

val_atoms = read("val.extxyz", ":")
e_fac, f_fac = compute_calibration_factors(calc, val_atoms, criterion="nll")   # or ma_cal, rms_cal, miscal
calc = ASECalculator("models/ens", transformations=[GlobalCalibration(energy_factor=e_fac, forces_factor=f_fac)])
```

Calibration only applies through `ASECalculator`. `apax md` dynamics checks use raw uncertainties: divide
your calibrated threshold by the factor.

## Use uncertainties in MD

- **Stop when unreliable:** `dynamics_checks` in the MD config ({doc}`md`). Start from 2–4× the 95th percentile of
  the raw force uncertainty on validation structures (see the loop in {doc}`active_learning`) and tune so MD at
  your target conditions triggers rarely.
- **Explore where the model is unsure** (uncertainty-driven dynamics) or **accelerate sampling** (GaMD), via ASE:

```python
from apax.md.function_transformations import UncertaintyDrivenDynamics
calc = ASECalculator("models/ens", transformations=[UncertaintyDrivenDynamics(height=0.1, width=0.05)])
```

Biased runs also report `energy_unbiased` and `forces_unbiased`. Structures flagged this way are candidates for
labelling: {doc}`active_learning`.
