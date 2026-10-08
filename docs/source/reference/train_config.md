(train_config)=

# Training config

Generated from the config classes (`apax.config.train_config`, `apax.config.model_config`), so defaults are exact.
Nested keys are written as `section.key`; sections with variants (e.g. `model`, `data.dataset`) are chosen by their
`name`/`processing`/`kind` key. Unknown keys are rejected. Templates: `apax template train` (minimal),
`apax template train --full` (all options; delete its `transfer_learning` section unless fine-tuning).

Minimal valid config:

```yaml
n_epochs: 100
data:
  directory: models
  experiment: my_model
  data_path: data.extxyz
  batch_size: 8
loss:
  - name: energy
  - name: forces
```

```{include} _generated/train_config_tables.md
```
