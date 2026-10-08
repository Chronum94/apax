# Evaluate a model

## Test-set errors

```yaml
data:
  test_data_path: test.extxyz    # recommended (without it the test set overlaps the validation set)
```

```bash
apax eval config.yaml             # uses the same training config
tail -1 models/apax/eval/log.csv  # last row = test-set result
```

Loads `best/`, predicts each test structure and writes `<model dir>/eval/log.csv`: the metric columns
(`test_energy_mae`, …) are running averages, so the last row is the result; `test_loss` is per structure.
Quirks (last structure skipped, file locations): {doc}`../reference/known_issues`.

## Your own analysis

```python
from ase.io import read
from apax.md import ASECalculator

calc = ASECalculator("models/apax")
ref = read("test.extxyz", ":")
pred = calc.batch_eval(ref, batch_size=64)          # copies with predictions attached (positions in Å)

import numpy as np
dE = [(p.get_potential_energy() - r.get_potential_energy()) / len(r) for p, r in zip(pred, ref)]
dF = np.concatenate([p.get_forces() - r.get_forces() for p, r in zip(pred, ref)])
print("E MAE per atom [meV]", 1000 * np.mean(np.abs(dE)), " F MAE [meV/Å]", 1000 * np.mean(np.abs(dF)))
```

## Is my model good enough?

Test errors are necessary, not sufficient. Rough targets for a single-system potential trained on DFT:

| Quantity | Typical good value |
|---|---|
| energy MAE per atom | 1–5 meV/atom (molecules < 1 kcal/mol per molecule) |
| force MAE | 30–100 meV/Å (≈ 5–15 % of the force RMS) |

Then check the application, not just the test set:
1. **Run the MD you care about** (temperature, duration) and check it stays stable ({doc}`md`).
2. **Re-label 20–50 MD snapshots with DFT** and evaluate on them: errors should match the test set. If they are much
   larger, the MD visits configurations missing from training → add them ({doc}`active_learning`).
3. **Compare observables** you know (lattice constant, RDF, density, phonons, barrier) with DFT/experiment.
4. **NVE drift** (MD with `ensemble: {name: nve}`): total energy should fluctuate, not drift.
5. **Ensemble disagreement** ({doc}`uncertainty`) flags unreliable regions without new DFT.

## Training curves

```python
import pandas as pd
log = pd.read_csv("models/apax/log.csv")
log.plot(x="epoch", y=["train_loss", "val_loss"], logy=True)
```
