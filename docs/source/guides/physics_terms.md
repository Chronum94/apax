# Repulsion terms and extra properties

## Short-range repulsion

Data rarely contains atoms closer than ~1 Å, so a plain network can predict nonsense (even attraction) at short range,
and high-temperature MD can collapse. Add a physical repulsion that the network corrects:

```yaml
model:
  name: gmnn          # always required when a model section is given
  empirical_corrections:
    - name: zbl          # Ziegler–Biersack–Littmark screened nuclear repulsion
      r_max: 1.5         # Å, smoothly switched off beyond
```

`exponential` (learnable exponential repulsion, same `r_max`) is the alternative. Their parameters train with
`optimizer.zbl_lr`, `rep_scale_lr`, `rep_prefactor_lr`.

## Predict extra properties (property heads)

Learn any per-atom or per-structure quantity from the same descriptor, e.g. partial charges or a dipole:

```yaml
model:
  name: gmnn
  property_heads:
    - name: charges         # must match the label key in your data and the loss/metric name
      aggregation: none     # none (per atom) | sum | mean (per structure)
      mode: l0              # l0 scalar | l1 vector | symmetric_l2 | symmetric_traceless_l2
      nn: [64, 64]
loss:
  - name: charges
    weight: 0.1
metrics:
  - name: charges
    reductions: [mae]
```

Labels are read from `atoms.calc.results[name]`, are not unit-converted, and must exist in every structure.
Predictions appear under the same name in `ASECalculator` results. With `n_shallow_members > 0` a head also predicts
`<name>_uncertainty` (a variance).
