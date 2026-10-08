# How apax works (5-minute mental model)

## The model

An apax model predicts one energy per atom from its neighbourhood within a cutoff `r_max` (default 5 Å) and sums them.
Forces (−∂E/∂R) and Hessians are derivatives of that energy, so MD conserves energy. Stress is also a derivative; see {doc}`../reference/known_issues` for current limits.

```
positions ─► neighbour list (r < r_max) ─► descriptor per atom ─► MLP readout ─► scale·h + shift (per element) ─► Σ atoms = E
                                                                                                    └─► autodiff: forces, stress, Hessian
```

- **Descriptor (GMNN, default):** radial basis functions of each neighbour distance, weighted by species-pair coefficients,
  combined into rotation-invariant Gaussian moments (up to 4-body). Strictly local, no message passing.
  Other models: `equiv-mp` (equivariant message passing), `so3krates` (attention).
- **Readout:** an MLP (`model.nn`, default `[256, 256]`) maps descriptors to an atomic energy.
- **Scale/shift:** per-element constants estimated from the training data before training (shift = per-element
  reference energies by ridge regression; scale = per-element force RMS), then fine-tuned. They make training start
  close to the right energy scale.

## The two configs

| File | Used by | Contains |
|---|---|---|
| training config (`config.yaml`) | `apax train`, `apax eval`, `apax md`, `ASECalculator` (reads it from the model dir) | data, model, loss, optimizer |
| MD config (`md_config.yaml`) | `apax md` | ensemble, temperature, timestep, duration, output |

Both are validated strictly: an unknown or misspelt key is an error, not a silent no-op. Check with `apax validate train|md FILE`.
Any omitted field takes its default; the trained model directory contains a `config.yaml` with every value filled in.

## The model directory

`data.directory/data.experiment` (e.g. `models/apax/`) is the model. Point `ASECalculator`, transfer learning and
other tools at this directory. It holds `best/` (lowest validation loss), `latest/` (for resuming), `log.csv`, the
expanded `config.yaml` and the train/validation split. Details: {doc}`../reference/files`.

## Units

Internally everything is eV and Å. Declare your data's units with `data.energy_unit` (eV, kcal/mol, Hartree, kJ/mol) and
`data.pos_unit` (Ang, Bohr); energies, forces and stresses are converted on load. Predictions are always in eV, eV/Å, eV/Å³.

## Compilation

JAX compiles the model the first time it sees an input shape. The first epoch, first MD block or first ASE call is
therefore slow (seconds to a minute); later calls are fast. apax pads structures and neighbour lists to fixed sizes to
avoid recompiling; a recompile shows up as a sudden slow step. See {doc}`../guides/performance`.

## Which interface for MD?

| Need | Use |
|---|---|
| Fast NVE/NVT/NPT on GPU, large systems | `apax md` (JAX-MD, fully compiled) — {doc}`../guides/md` |
| Anything ASE can do (relax, NEB, phonons, custom MD) | `ASECalculator` — {doc}`../guides/ase` |
| Biomolecular setups, OpenMM integrators/reporters | OpenMM interface — {doc}`../guides/openmm` |
