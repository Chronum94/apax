import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from vesin import NeighborList

from apax.data.input_pipeline import BatchProcessor, pad_nl
from apax.data.preprocessing import compute_nl
from apax.utils.convert import atoms_to_inputs

# The descriptor's segment sums are flagged `indices_are_sorted`, which XLA does
# not check: every neighbour list given to the model must be sorted by centre
# (row 1), including its padding.


def is_centre_sorted(idx):
    return bool(np.all(np.diff(idx[1].astype(np.int64)) >= 0))


def pair_set(idx, offsets, positions):
    """Pairs as (i, j, displacement) so that a reordering is detectable."""
    dr = positions[idx[1]] - positions[idx[0]] + offsets
    return sorted(
        (int(i), int(j), *np.round(d, 6)) for i, j, d in zip(idx[0], idx[1], dr)
    )


def make_structures():
    periodic = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    periodic.rattle(0.1, seed=1)
    small = bulk("Si", "diamond", a=5.43)  # many periodic images within the cutoff
    small.rattle(0.05, seed=2)
    for atoms in (periodic, small):
        atoms.wrap()  # apax works with wrapped (fractional) positions
    return [periodic, small]


@pytest.mark.parametrize("atoms", make_structures() + [molecule("C6H6")])
def test_compute_nl_is_centre_sorted_and_complete(atoms):
    r_max = 5.0
    inputs = atoms_to_inputs([atoms])
    positions, box = inputs["positions"][0], inputs["box"][0]
    idx, offsets = compute_nl(positions, box, r_max)
    assert is_centre_sorted(idx)

    periodic = bool(np.any(atoms.pbc))
    if periodic:
        i, j, S = NeighborList(cutoff=r_max, full_list=True).compute(
            points=atoms.positions, box=atoms.cell.array, periodic=True, quantities="ijS"
        )
        ref_offsets = S @ atoms.cell.array
    else:
        i, j = NeighborList(cutoff=r_max, full_list=True).compute(
            points=atoms.positions, box=np.zeros((3, 3)), periodic=False, quantities="ij"
        )
        ref_offsets = np.zeros((len(i), 3))
    cart = atoms.positions
    ref = pair_set(np.array([i, j]), ref_offsets, cart)
    assert pair_set(idx, offsets, cart) == ref


def test_padding_keeps_order():
    atoms = make_structures()[0]
    inputs = atoms_to_inputs([atoms])
    idx, offsets = compute_nl(inputs["positions"][0], inputs["box"][0], 5.0)
    padded, padded_offsets = pad_nl(idx, offsets, idx.shape[1] + 37)
    assert padded.shape[1] == idx.shape[1] + 37
    assert is_centre_sorted(padded)
    # padding pairs are self-pairs with zero offset, masked like the old (0, 0) ones
    assert np.all(padded[0, idx.shape[1] :] == padded[1, idx.shape[1] :])
    assert np.all(padded_offsets[idx.shape[1] :] == 0)


def test_batch_processor_keeps_order():
    structures = make_structures()
    inputs = atoms_to_inputs(structures)
    samples = []
    for k, atoms in enumerate(structures):
        inp = {key: inputs[key][k] for key in ("positions", "box", "numbers", "n_atoms")}
        samples.append((inp, {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}))
    batch, _ = BatchProcessor(cutoff=5.0, atom_padding=8, nl_padding=64)(samples)
    for k in range(len(structures)):
        assert is_centre_sorted(batch["idx"][k])


def test_empty_neighbor_list_pads():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 20.0]])
    inputs = atoms_to_inputs([atoms])
    idx, offsets = compute_nl(inputs["positions"][0], inputs["box"][0], 3.0)
    padded, _ = pad_nl(idx, offsets, 4)
    assert padded.shape == (2, 4) and is_centre_sorted(padded)
