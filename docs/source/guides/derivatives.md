# Stress, Hessians and vibrations

Forces, stress and Hessians are derivatives of the model energy, so a model trained on energies and forces can
predict them (stress: within the limits below).

## Stress

:::{danger}
In apax 0.13 stress is wrong during **training** (do not train on stress) and at inference for cells thinner than
2·`r_max`. Inference on larger cells is correct: replicate small cells (`atoms.repeat`) before cell relaxation or
NPT ({doc}`../reference/known_issues`).
:::

```yaml
model:
  name: gmnn
  calc_stress: true        # needed to train on stress
loss:
  - name: stress
    loss_type: tril        # lower triangle of the symmetric tensor; computed on stress × volume (eV)
    weight: 0.1
```

Inference: `ASECalculator(model_dir, calc_stress=True)` → `atoms.get_stress()` (eV/Å³, ASE convention). Off by
default unless the model was trained with `calc_stress: true`.

## Hessians and frequencies

```python
from ase.vibrations import VibrationsData
from apax.md import ASECalculator

calc = ASECalculator("models/apax", calc_hessian=True)     # required for get_hessian
atoms.calc = calc
H = calc.get_hessian(atoms)                                 # (3N, 3N), eV/Å²
freqs = VibrationsData.from_2d(atoms, H).get_frequencies() # cm⁻¹
```

Use `VibrationsData` (analytic Hessian), not `ase.vibrations.Vibrations` (finite differences of forces). Relax the
structure first.

## Training on Hessians

Data: a `hessian` label of shape (N, 3, N, 3) per structure (H5MD recommended), in eV/Å² (not unit-converted).

1. Train on energies and forces until converged.
2. Fine-tune with a small mass-weighted Hessian loss:

```yaml
model:
  name: gmnn
  calc_hessian: true
loss:
  - name: energy
  - name: forces
    weight: 4.0
  - name: hessian
    loss_type: mw_hessian    # divides by sqrt(m_i m_j): balances modes
    weight: 0.01             # keep low; Hessian elements are large
transfer_learning:
  base_model_checkpoint: models/ef_model
```

Hessian training is expensive (second derivatives): use small batches. If forces degrade, lower the Hessian weight or
increase capacity (`n_radial`, `nn`).
