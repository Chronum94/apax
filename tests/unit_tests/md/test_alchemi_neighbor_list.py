"""ALCHEMI neighbour-list adapter (md/alchemi_neighbor_list.py).

GPU (+ nvalchemi-toolkit-ops): identical pair sets, layout, padding and capacity vs jax-md's
Sparse list on cubic and triclinic boxes; the cached reverse-pair row; the skin rebuild
criterion; Morton-sorted lists map back to the same pairs.
CPU: the species-preserving Morton permutation and the sorted_frame model wrapper."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk
from jax_md import partition

from apax.md.alchemi_neighbor_list import morton_order, sorted_frame
from apax.utils.space import periodic_general_elementwise

CUTOFF, SKIN, CM, EXTRA = 5.0, 0.5, 1.25, 8


def _atoms(sheared):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(
        (5, 5, 5)
    )  # 1000 atoms, 28.2 A box
    if sheared:
        cell = atoms.cell.array.copy()
        cell[1] += 0.15 * cell[0]
        cell[2] += 0.10 * cell[0] + 0.12 * cell[1]
        atoms.set_cell(cell, scale_atoms=True)
    atoms.rattle(0.15, seed=2)
    rng = np.random.default_rng(5)
    return atoms[rng.permutation(len(atoms))]  # scrambled order, as after a long MD run


def _pairs(idx, N, perm=None):
    idx = np.asarray(idx)
    v = idx[1] < N
    c, n = idx[1][v], idx[0][v]
    if perm is not None:
        c, n = perm[c], perm[n]
    return set(zip(c.tolist(), n.tolist()))


# ------------------------------------------------------------------ CPU: sorting utilities


def test_morton_order_is_species_preserving_permutation():
    atoms = _atoms(False)
    frac = jnp.asarray(atoms.get_scaled_positions(wrap=True), jnp.float32)
    Z = atoms.numbers
    perm, inv = (np.asarray(x) for x in morton_order(frac, Z))
    N = len(Z)
    np.testing.assert_array_equal(np.sort(perm), np.arange(N))
    np.testing.assert_array_equal(perm[inv], np.arange(N))
    np.testing.assert_array_equal(Z[perm], Z)  # the model can keep its constant Z
    # spatial locality improves on a scrambled input: neighbours get closer indices
    d = frac[:, None, :] - frac[None, :, :]
    d = np.asarray(d - jnp.round(d)) @ atoms.cell.array
    dist = np.linalg.norm(d, axis=-1)
    i, j = np.nonzero((dist < 4.0) & (dist > 0.0))  # neighbour pairs, no self pairs
    gap_before = np.abs(i - j).mean()
    gap_after = np.abs(inv[i] - inv[j]).mean()
    assert gap_after < 0.5 * gap_before


def test_sorted_frame_evaluates_in_sorted_space_and_maps_back():
    rng = np.random.default_rng(0)
    N = 64
    Z = np.repeat([1, 8], N // 2)
    R = jnp.asarray(rng.normal(size=(N, 3)))
    perm = np.concatenate([rng.permutation(N // 2), N // 2 + rng.permutation(N // 2)])
    inv = np.argsort(perm)
    assert np.array_equal(Z[perm], Z)

    def model(params, R, Z, neighbor, box, offsets):
        assert neighbor is idx  # the wrapper passes the (sorted-space) index array
        return {
            "energy": jnp.sum(Z * R[:, 0] ** 2),
            "forces": R * Z[:, None],
            "stress": jnp.eye(3),
        }

    idx = jnp.zeros((2, 4), jnp.int32)
    nbrs = types.SimpleNamespace(idx=idx, perm=jnp.asarray(perm), inv=jnp.asarray(inv))
    out = sorted_frame(model, unpermute_outputs=True)(
        None, R, jnp.asarray(Z), nbrs, None, None
    )
    np.testing.assert_allclose(out["energy"], jnp.sum(Z * R[:, 0] ** 2), rtol=1e-12)
    np.testing.assert_allclose(out["forces"], R * Z[:, None], rtol=1e-12)
    np.testing.assert_array_equal(out["stress"], jnp.eye(3))
    # without a permutation the wrapper is the identity
    plain = types.SimpleNamespace(idx=idx)
    out2 = sorted_frame(lambda *a: a[3], unpermute_outputs=True)(
        None, R, Z, plain, None, None
    )
    assert out2 is plain


# ------------------------------------------------------------------ GPU: ALCHEMI lists

gpu = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="needs a CUDA GPU"
)


@pytest.fixture(scope="module")
def alchemi():
    pytest.importorskip("nvalchemiops")
    from apax.md.alchemi_neighbor_list import alchemi_neighbor_list

    return alchemi_neighbor_list


def _reference(atoms):
    box = np.asarray(atoms.cell.array.T)
    disp, _ = periodic_general_elementwise(box)
    fn = partition.neighbor_list(
        disp,
        box,
        CUTOFF,
        SKIN,
        fractional_coordinates=True,
        format=partition.Sparse,
        capacity_multiplier=CM,
    )
    frac = jnp.asarray(atoms.get_scaled_positions(wrap=True), jnp.float32)
    return box, frac, fn.allocate(frac, extra_capacity=EXTRA)


@gpu
@pytest.mark.parametrize("sheared", [False, True], ids=["cubic", "triclinic"])
def test_alchemi_list_matches_jax_md(alchemi, sheared):
    atoms = _atoms(sheared)
    N = len(atoms)
    box, frac, ref = _reference(atoms)
    new = alchemi(box, CUTOFF, SKIN, CM).allocate(frac, extra_capacity=EXTRA)
    b = np.asarray(new.idx)
    P = b.shape[1]
    assert not bool(new.did_buffer_overflow)
    assert _pairs(b, N) == _pairs(ref.idx, N)
    assert P == ref.idx.shape[1]  # same capacity rule as jax-md
    assert np.all(np.diff(b[1]) >= 0)  # centre-sorted
    n_real = int((b[1] < N).sum())
    assert np.all(b[:2, n_real:] == N)  # padding
    # row 2: reverse slot of every pair, P for padding
    assert b.shape[0] == 3
    rv, v = b[2], b[1] < N
    assert np.all(rv[~v] == P)
    np.testing.assert_array_equal(b[0][rv[v]], b[1][v])
    np.testing.assert_array_equal(b[1][rv[v]], b[0][v])


@gpu
def test_alchemi_rebuild_criterion(alchemi):
    atoms = _atoms(False)
    box, frac, _ = _reference(atoms)
    nbrs = alchemi(box, CUTOFF, SKIN, CM).allocate(frac, extra_capacity=EXTRA)
    upd = jax.jit(lambda pos, nb: nb.update(pos))
    small = upd(frac + 1e-4, nbrs)  # << skin / 2
    large = upd(frac + 0.03, nbrs)  # 0.03 x 28 A > skin / 2
    assert bool(jnp.all(small.reference_position == nbrs.reference_position))
    assert not bool(jnp.all(large.reference_position == nbrs.reference_position))
    assert not bool(large.did_buffer_overflow)


@gpu
def test_alchemi_sorted_list_maps_back(alchemi):
    atoms = _atoms(True)
    N = len(atoms)
    box, frac, ref = _reference(atoms)
    fn = alchemi(box, CUTOFF, SKIN, CM, sort_atoms=True, species=atoms.numbers)
    nbrs = fn.allocate(frac, extra_capacity=EXTRA)
    perm = np.asarray(nbrs.perm)
    np.testing.assert_array_equal(atoms.numbers[perm], atoms.numbers)
    # sorted slot s is atom perm[s]: mapping both ends back gives jax-md's pair set
    assert _pairs(nbrs.idx, N, perm) == _pairs(ref.idx, N)
