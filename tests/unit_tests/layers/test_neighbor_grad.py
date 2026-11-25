"""Warp pair-structure kernels (neighbor_grad.py): CSR segment starts, reverse-pair map,
the scatter-free gather backward (gather_pair) and the fused periodic pair vectors
(pair_vectors), against host / XLA references on jax-md sparse lists of cubic and
triclinic boxes and on synthetic edge cases."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk
from jax_md import partition

pytest.importorskip("warp")

from apax.layers.neighbor_grad import (  # noqa: E402
    centre_csr,
    gather_pair,
    pair_vectors,
    reverse_pairs,
)
from apax.utils.space import periodic_general_elementwise  # noqa: E402

pytestmark = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="needs a CUDA GPU"
)

CUTOFF, SKIN = 5.0, 0.5


def _system(sheared):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(
        (5, 5, 5)
    )  # 1000 atoms, 28.2 A box
    if sheared:
        cell = atoms.cell.array.copy()
        cell[1] += 0.15 * cell[0]
        cell[2] += 0.10 * cell[0] + 0.12 * cell[1]
        atoms.set_cell(cell, scale_atoms=True)
    atoms.rattle(0.15, seed=2)
    box = np.asarray(atoms.cell.array.T)  # lattice vectors as columns (apax convention)
    disp, _ = periodic_general_elementwise(box)
    nbr_fn = partition.neighbor_list(
        disp, box, CUTOFF, SKIN, fractional_coordinates=True, format=partition.Sparse
    )
    frac = jnp.asarray(atoms.get_scaled_positions(wrap=True), jnp.float32)
    idx = nbr_fn.allocate(frac, extra_capacity=8).idx
    return frac, idx, box, disp, len(atoms)


def _starts_reference(centre, n_atoms):
    centre = np.array(centre)
    if len(centre) > 1 and centre[-1] < centre[-2]:
        centre[-1] = n_atoms  # jax-md's stray last slot counts as padding
    counts = np.bincount(np.minimum(centre, n_atoms), minlength=n_atoms + 1)[:n_atoms]
    return np.concatenate([[0], np.cumsum(counts)])


def _synthetic_lists():
    rng = np.random.default_rng(0)
    c = np.sort(rng.integers(0, 50, 400))
    c[-30:] = 50  # padding at the end and atoms without pairs
    yield "padding", np.stack([rng.integers(0, 50, 400), c]), 50
    yield (
        "no_padding",
        np.stack([rng.integers(0, 50, 400), np.sort(rng.integers(0, 50, 400))]),
        50,
    )
    c3 = np.sort(np.where(np.arange(64) < 10, 3, 20))
    c3[-1] = 1  # stray out-of-order last slot
    yield "stray_slot", np.stack([np.zeros(64, int), c3]), 20


@pytest.mark.parametrize("case", list(_synthetic_lists()), ids=lambda c: c[0])
def test_centre_csr_synthetic(case):
    _, idx, n = case
    _, starts = jax.jit(lambda i: centre_csr(i, n))(jnp.asarray(idx, jnp.int32))
    np.testing.assert_array_equal(np.asarray(starts), _starts_reference(idx[1], n))


@pytest.mark.parametrize("sheared", [False, True], ids=["cubic", "triclinic"])
def test_csr_and_reverse_pairs_on_jax_md_list(sheared):
    _, idx, _, _, N = _system(sheared)
    idx_np = np.asarray(idx)
    P = idx_np.shape[1]
    rev, pairs, starts = jax.jit(lambda i: reverse_pairs(i, N, with_starts=True))(idx)
    np.testing.assert_array_equal(np.asarray(starts), _starts_reference(idx_np[1], N))
    slot = {
        (int(idx_np[1, p]), int(idx_np[0, p])): p for p in range(P) if idx_np[1, p] < N
    }
    ref = np.array(
        [
            slot.get((int(idx_np[0, p]), int(idx_np[1, p])), P) if idx_np[1, p] < N else P
            for p in range(P)
        ]
    )
    np.testing.assert_array_equal(np.asarray(rev), ref)
    # a cached third row (as written by the ALCHEMI adapter) is used as is
    rev3, _ = reverse_pairs(jnp.concatenate([idx, jnp.asarray(ref, idx.dtype)[None]]), N)
    np.testing.assert_array_equal(np.asarray(rev3), ref)


@pytest.mark.parametrize("sheared", [False, True], ids=["cubic", "triclinic"])
def test_gather_pair_vjp(sheared):
    R, idx, _, _, N = _system(sheared)

    def plain(R):
        return jnp.take(R, idx[1], axis=0, mode="clip"), R[jnp.minimum(idx[0], N - 1)]

    valid = (idx[1] < N)[:, None]
    rng = np.random.default_rng(3)
    cts = tuple(
        jnp.asarray(rng.normal(size=(idx.shape[1], 3)), jnp.float32) * valid
        for _ in range(2)
    )
    a, va = jax.vjp(plain, R)
    b, vb = jax.vjp(lambda R: gather_pair(R, idx), R)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(
            np.asarray(x)[np.asarray(valid[:, 0])], np.asarray(y)[np.asarray(valid[:, 0])]
        )
    (ga,), (gb,) = va(cts), vb(cts)
    assert float(jnp.abs(ga - gb).max() / jnp.abs(ga).max()) < 1e-5


@pytest.mark.parametrize("sheared", [False, True], ids=["cubic", "triclinic"])
@pytest.mark.parametrize("cached_rev", [False, True])
def test_pair_vectors_match_xla(sheared, cached_rev):
    R, idx, box, disp, N = _system(sheared)
    box32 = jnp.asarray(box, jnp.float32)
    if cached_rev:
        rev, _ = reverse_pairs(idx, N)
        idx = jnp.concatenate([idx, rev[None].astype(idx.dtype)])

    def reference(R):
        Rj, Ri = gather_pair(R, idx[:2])
        return jax.vmap(lambda a, b: disp(a, b, box=box32))(Rj, Ri)

    a, va = jax.vjp(reference, R)
    b, vb = jax.vjp(lambda R: pair_vectors(R, idx, box32), R)
    # fp32 wrap / box-product rounding: 2-3e-6 A measured
    assert float(jnp.abs(a - b).max()) < 2e-5
    ct = jnp.asarray(np.random.default_rng(4).normal(size=a.shape), jnp.float32)
    ct = ct * (idx[1] < N)[:, None]  # padding cotangents are zero in the model (masked)
    (ga,), (gb,) = va(ct), vb(ct)
    np.testing.assert_allclose(
        np.asarray(gb), np.asarray(ga), rtol=1e-5, atol=1e-5 * float(jnp.abs(ga).max())
    )
