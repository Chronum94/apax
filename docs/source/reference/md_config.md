(md_config)=

# MD config

Generated from `apax.config.md_config`. Units: time fs, temperature K, pressure bar, distance Å, energy eV.
Template: `apax template md`. Guide: {doc}`../guides/md`.

Minimal valid config:

```yaml
ensemble:
  name: nvt
  dt: 0.5
  temperature_schedule: {name: constant, T0: 300}
duration: 10000
initial_structure: start.extxyz
```

Notes on behaviour that the field descriptions don't capture:
- `ensemble.thermostat_chain` is currently ignored for NVT (defaults are used).
- `checkpoint_interval` counts blocks of `n_inner` steps.
- `initial_structure` uses the **last** frame of the file.
- `properties` doubles as a switch: listed derivatives (e.g. `stress`) are computed.

```{include} _generated/md_config_tables.md
```
