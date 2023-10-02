import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from apax.data.statistics import PerElementRegressionShift


def test_energy_per_element():
    pos = [0.0, 0.0, 0.0]
    atoms1 = Atoms("H2O", positions=[pos, pos, pos])
    atoms2 = Atoms("CH4", positions=[pos, pos, pos, pos, pos])
    atoms3 = Atoms("N2", positions=[pos, pos])

    dummy_energies = np.array([0, 20.0, 0, 0, 0, 0, 50.0, 100.0, 70.0])

    atoms_list = [atoms1, atoms2, atoms3]
    energies = []
    n_atoms = []
    for atoms in atoms_list:
        n_atoms.append(len(atoms))
        print(len(atoms))
        energy = np.sum(dummy_energies[atoms.numbers])
        energies.append(energy)
        atoms.calc = SinglePointCalculator(atoms, energy=energy)

    labels = {
        "ragged": {
            "energy": [atoms.get_potential_energy() for atoms in atoms_list],
        }
    }
    inputs = {
        "ragged": {
            "numbers": [atoms.numbers for atoms in atoms_list],
        },
        "fixed": {
            "n_atoms": n_atoms,
        },
    }

    elemental_shift = PerElementRegressionShift.compute(
        inputs, labels, {"energy_regularisation": 0.0}
    )
    regression_energies = []
    for atoms in atoms_list:
        energy = np.sum(elemental_shift[atoms.numbers])
        regression_energies.append(energy)

    assert np.allclose(energies, regression_energies, atol=1e-4)
