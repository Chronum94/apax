# Fine-tune a trained model (transfer learning)

Two common cases:

| Case | Reset `scale_shift`? | Learning rates |
|---|---|---|
| new data at the **same** level of theory (e.g. an active-learning iteration) | no | ~0.1–0.3× the original |
| **different** level of theory (e.g. DFT → CCSD(T)), few new labels | yes (new reference energies) | low; optionally freeze the descriptor |

Same level of theory (e.g. an active-learning iteration):

```yaml
transfer_learning:
  base_model_checkpoint: models/iter0     # model directory of the base model
optimizer:                                # ~0.1–0.3× the defaults
  emb_lr: 2.0e-4
  nn_lr: 2.0e-4
  scale_lr: 2.0e-5
  shift_lr: 5.0e-4
```

New level of theory with few labels (e.g. DFT → CCSD(T)):

```yaml
transfer_learning:
  base_model_checkpoint: models/dft_model
  reset_layers: [scale_shift]             # new reference energies
  freeze_layers: [representation]         # optional: keep the descriptor, retrain readout + scale/shift
optimizer:
  nn_lr: 1.0e-4
  scale_lr: 1.0e-4
  shift_lr: 1.0e-3
```

- **Architecture must match** the base model: copy its `model:` section from `models/pretrained/config.yaml`.
- `reset_layers`: exact module names that keep their fresh initialisation instead of the pretrained values
  (`scale_shift`, `dense_0`, `dense_1`, …).
- `freeze_layers`: substrings of parameter paths whose parameters are not updated. Paths look like
  `energy_model/representation/...` (descriptor, including species/radial coefficients), `energy_model/readout/dense_i/...`,
  `energy_model/scale_shift/...`. `apax visualize config.yaml` prints the module tree.
- Alternatively freeze a parameter group with a learning rate ≤ 1e-7 (`emb_lr: 0.0`, `nn_lr`, `scale_lr`, `shift_lr`, `zbl_lr`).

:::{warning}
Only add a `transfer_learning` section when fine-tuning: its presence enables transfer learning even if
`base_model_checkpoint` is `null`, and training crashes.
:::

Worked example (benzene DFT → CCSD(T)): notebook 03 in {doc}`../_tutorials/index`.
