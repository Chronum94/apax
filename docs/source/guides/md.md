# Run MD with JAX-MD (`apax md`)

```bash
apax template md                        # writes md_config.yaml
apax validate md md_config.yaml
apax md config.yaml md_config.yaml      # training config (finds the model) + MD config
```

The model is loaded from `<directory>/<experiment>/best` of the **training** config. Python:
`from apax.md import run_md; run_md("config.yaml", "md_config.yaml")`. Runs on one GPU.

## Minimal MD config

```yaml
ensemble:
  name: nvt                  # nve | nvt | npt
  dt: 0.5                    # fs
  temperature_schedule:
    name: constant
    T0: 300                  # K
duration: 100000             # fs (100 ps)
initial_structure: start.extxyz   # the LAST frame of this file is used
sim_dir: md
```

:::{important}
Every cell height must be ≥ 2·`r_max` (10 Å for the default 5 Å). Smaller boxes only log
`cutoff larger than half the box` and continue with **wrong energies**. Replicate the cell (`atoms.repeat`) first.
:::

## Ensembles

| `ensemble.name` | Extra fields | Notes |
|---|---|---|
| `nve` | — | `T0` only sets initial velocities |
| `nvt` | — | Nosé–Hoover chain; `thermostat_chain` settings are currently ignored |
| `npt` | `pressure` (bar, default 1.01325), `barostat_chain: {tau: 1000}` | cells ≥ 2·`r_max` in every direction (replicate typical ~7 Å DFT cells first, {doc}`../reference/known_issues`); set `extra_capacity`; no constraints |

`tau` values are in time steps (multiples of `dt`). Temperature schedules: `constant` (`T0`), `piecewise`
(`T0`, `temperatures: [...]`, `durations: [...]` in steps), `oscillating_ramp` (`T0`, `Tend`, `amplitude`,
`num_oscillations`, `total_steps`).

## Output

| Setting | Default | Meaning |
|---|---|---|
| `sim_dir` | `.` | output directory |
| `traj_name` | `md.h5` | H5MD trajectory |
| `sampling_rate` | 10 | save every N time steps |
| `buffer_size` | 2500 | frames held in memory before writing to disk |
| `properties` | energy, forces, stress, uncertainties, … | what to save; listing `stress` makes MD compute it — drop it when not needed |
| `wrapped` | true | wrap positions into the cell (unwrap for diffusion) |

Also written: `md.log` (ends with a ns/day and µs/step/atom summary) and `ckpts/` (restart checkpoints).

Reading the trajectory:

```python
import numpy as np, znh5md
traj = znh5md.IO("md/md.h5")                 # list-like of ase.Atoms
frames = traj[::10]                          # slicing reads only what you need
E = np.array([a.get_potential_energy() for a in traj])
from ase.io import write; write("md.extxyz", traj[:])   # e.g. for OVITO / VMD
```

Stored stress is stress × volume (eV) and `time`/`step` are frame index × `dt` (multiply by `sampling_rate`):
{doc}`../reference/known_issues`.

## Restart and long runs

- `restart: true` (default) continues from `sim_dir/ckpts`. To extend a finished run, increase `duration` and re-run
  the same command. Delete `sim_dir` to start over.
- `checkpoint_interval` (default 50 000) counts blocks of `n_inner` steps, i.e. 50 000 × 500 steps by default:
  set it so a checkpoint is written every ~30 min of wall time.
- Frames reach disk every `buffer_size` frames (default 2500); a job killed earlier loses them. Keep it moderate
  (100–1000).
- Prefer finishing and extending runs (raise `duration`) over killing them: restarts after a kill can duplicate frames
  ({doc}`../reference/known_issues`).

SLURM sketch:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --time=24:00:00
grep -q "dynamics checks failed" md/md.log && exit 1   # stopped by a check: inspect first
apax md config.yaml md_config.yaml     # resubmit the same script to continue (compiles at each start)
```

## Safety, constraints, biases

```yaml
dynamics_checks:                       # stop the run when exceeded
  - name: forces_uncertainty           # needs an ensemble model
    threshold: 1.0                     # eV/Å (raw, uncalibrated uncertainty)
  - name: energy_uncertainty           # needs an ensemble model
    threshold: 0.05                    # eV/atom (per_atom: true)
  - name: reflection                   # any model; orthorhombic periodic cells
    cutoff_plane_height: 30.0          # stop if an atom crosses z = 30 Å
constraints:
  - name: fixatoms
    indices: [0, 1, 2]
  - name: fixcenterofmass
    position: initial                  # or origin, or [x, y, z]
  - name: fixlayer
    lower_limit: 0.0                   # fix atoms with lower ≤ z ≤ upper (Å)
    upper_limit: 5.0
biases:
  - name: spherical_wall
    radius: 15.0                       # Å
    spring_constant: 1.0               # eV/Å²
```

Checks are evaluated on saved frames and stop the run at the end of the current `n_inner` block, so the trajectory
ends slightly past the trigger and the exit code is 0; look for `One or more dynamics checks failed` in `md.log`. Re-running a stopped run
restarts from the last checkpoint and stops again: change the setup or delete `ckpts/` first.

## Performance knobs

| Setting | Default | Effect |
|---|---|---|
| `n_inner` | 500 | steps per compiled block; larger = less overhead, coarser checks/checkpoints |
| `dr_threshold` | 0.5 Å | neighbour-list skin; larger = fewer rebuilds, more pairs |
| `extra_capacity` | 0 | spare neighbour slots; set > 0 when density changes (NPT, heating); reallocation can crash |
| `disable_cell_list` | false | true for non-periodic systems (no cell list needed) |

The run is rounded up to whole `n_inner` blocks.
