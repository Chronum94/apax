# Workflows: IPSuite/zntrack nodes and hyperparameter search

`pip install "apax[ipsuite]"` provides zntrack nodes in `apax.nodes` for reproducible, DVC-tracked pipelines
([IPSuite](https://github.com/zincware/ipsuite)).

| Node | Does |
|---|---|
| `Apax(data, validation_data, config, model=None)` | train (`model=` a base model → fine-tune) |
| `ApaxEnsemble(models)` | combine trained nodes into an ensemble calculator |
| `ApaxJaxMD(data, model, config)` | `apax md` on a structure from `data` |
| `ApaxBatchPrediction(data, model)` | predict a dataset |
| `BatchKernelSelection(data, train_data, models, n_configurations)` | active-learning selection ({doc}`active_learning`) |
| `ApaxCalibrate(model, validation_data)` | uncertainty calibration |
| `ApaxApplyTransformation(model, transformations)` | add UDD/GaMD biases |
| `ApaxImport(config)` | wrap an existing model directory |
| `ApaxOptimizeHyperparameters(config, data, validation_data, optuna_config_path)` | Optuna search (`apax[optuna]`) |

Node configs are normal training configs; the node sets `data.directory`, `experiment` and the data paths itself.
Validate such a template with `apax validate train config.yaml --template`.

```python
import ipsuite as ips
import apax.nodes

with ips.Project() as project:
    data = ips.AddData(file="train.extxyz")
    model = apax.nodes.Apax(data=data.frames, validation_data=data.frames, config="config.yaml")
project.repro()
```
