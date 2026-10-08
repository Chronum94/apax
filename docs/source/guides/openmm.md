# Use apax from OpenMM

Requires `pip install "apax[openmm]"` (OpenMM ≥ 8.5). Units (eV/Å ↔ kJ/mol/nm) are converted for you.

```python
from ase.io import read
from openmm import LangevinMiddleIntegrator, unit
from apax.md.openmm_interface import create_system, create_simulation, get_PythonForce_from_Apax

atoms = read("start.extxyz")
system = create_system(atoms)                                  # particles + masses (+ box)
system.addForce(get_PythonForce_from_Apax("models/apax", atoms))
integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.5 * unit.femtoseconds)
sim = create_simulation(atoms, system, integrator)
sim.step(1000)
```

Write trajectories with standard OpenMM reporters or `apax.utils.openmm_reporters.XYZReporter`
(extended XYZ, optional velocities/forces). Notebook 02 in {doc}`../_tutorials/index` has a complete example.

For throughput, `apax md` ({doc}`md`) avoids the per-step Python callback and is usually faster.
