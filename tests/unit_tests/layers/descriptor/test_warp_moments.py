"""Warp Gaussian-moment kernels (warp_moments.py) vs an XLA fp64 reference.

Packed moments m0 | m1 | p2 | p3 and their VJP w.r.t. the pair vectors, on a jax-md sparse
neighbour list of a rattled two-element crystal, with a random per-pair-type radial table
and a random radial compression."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk
from jax_md import partition, space

pytest.importorskip("warp")

from apax.layers.descriptor.basis_functions import BesselBasis, cosine_cutoff  # noqa: E402
from apax.layers.descriptor.moments import _U2, _U3  # noqa: E402
from apax.layers.descriptor.warp_moments import (  # noqa: E402
    default_split,
    fold_compression,
    warp_packed_moments,
)
from apax.layers.masking import mask_by_neighbor  # noqa: E402
from apax.utils.space import distance  # noqa: E402

requires_gpu = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="needs a CUDA GPU"
)

R_MAX, NB, K, C = 5.0, 12, 12, 5


def _system():
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(
        (4, 4, 4)
    )  # 512 atoms, 22.6 A box
    atoms.rattle(0.15, seed=1)
    box = jnp.asarray(atoms.cell.array.T)
    disp, _ = space.periodic_general(box, fractional_coordinates=False)
    nbr_fn = partition.neighbor_list(
        disp, box, R_MAX, 0.5, fractional_coordinates=False, format=partition.Sparse
    )
    atoms.wrap()
    pos = jnp.asarray(atoms.positions, jnp.float32)
    idx = nbr_fn.allocate(pos, extra_capacity=16).idx
    N = len(atoms)
    dr = jax.vmap(disp)(pos[jnp.minimum(idx[1], N - 1)], pos[jnp.minimum(idx[0], N - 1)])
    dr = jnp.where((idx[1] < N)[:, None], dr, 0.0).astype(
        jnp.float32
    )  # centre - neighbour
    return atoms, idx, dr


def _reference(dr, idx, Z, smap, n_el, T0, W, N):
    dr = dr.astype(jnp.float64)
    r = distance(dr)  # NaN-safe at the zero vectors of padded pairs, as the model
    dn = dr / (r + 1e-5)[:, None]
    zi = Z[jnp.minimum(idx[0], N - 1)]
    zj = Z[jnp.minimum(idx[1], N - 1)]
    t = smap[zj] * n_el + smap[zi]  # pair type: centre-major
    basis = BesselBasis(n_basis=NB, r_max=R_MAX, dtype=jnp.float64).apply({}, r)
    rad = jnp.einsum("pb,pbk->pk", basis, T0[t]) * cosine_cutoff(r, R_MAX)[:, None]
    rad = mask_by_neighbor(rad, idx, dr)
    rad = jnp.where((idx[1] < N)[:, None], rad, 0.0)
    seg = jnp.minimum(idx[1], N)
    rt = rad @ W
    u2 = jnp.stack([dn[:, a] * dn[:, b] for a, b in _U2], -1)
    u3 = jnp.stack([dn[:, a] * dn[:, b] * dn[:, c] for a, b, c in _U3], -1)

    def ssum(x):
        return jax.ops.segment_sum(x, seg, N + 1)[:N].reshape(N, -1)

    return jnp.concatenate(
        [
            ssum(rad),
            ssum(rt[:, :, None] * dn[:, None, :]),
            ssum(rt[:, :, None] * u2[:, None, :]),
            ssum(rt[:, :, None] * u3[:, None, :]),
        ],
        axis=1,
    )


def test_default_split_heuristic():
    # refit on an L40S (E24): 4 at 3.9k atoms, 2 at 17.9k, 1 at >= 50k
    assert [default_split(n, C) for n in (3861, 17875, 104247)] == [4, 2, 1]
    assert all(1 <= default_split(n, C) <= 16 for n in (1, 10, 10**7))


@requires_gpu
@pytest.mark.parametrize("n_split", [None, 1, 8])
def test_warp_moments_match_xla_fp64(n_split):
    atoms, idx, dr = _system()
    N = len(atoms)
    Z = jnp.asarray(atoms.numbers)
    els = sorted(set(atoms.numbers.tolist()))
    n_el = len(els)
    smap = np.zeros(119, np.int32)
    smap[els] = np.arange(n_el)
    rng = np.random.default_rng(0)
    T0 = rng.normal(size=(n_el * n_el, NB, K)) * 0.3
    W = rng.normal(size=(K, C)) * 0.3

    ref, vjp_ref = jax.vjp(
        lambda d: _reference(
            d, idx, Z, jnp.asarray(smap), n_el, jnp.asarray(T0), jnp.asarray(W), N
        ),
        dr,
    )

    def warp(d):
        T0f = jnp.asarray(T0, jnp.float32)
        T1f = fold_compression(T0f, jnp.asarray(W, jnp.float32))
        m0, m1, p2, p3 = warp_packed_moments(
            d, Z, idx, smap, T0f, T1f, R_MAX, n_split=n_split
        )
        return jnp.concatenate(
            [m0, m1.reshape(N, -1), p2.reshape(N, -1), p3.reshape(N, -1)], axis=1
        )

    out, vjp_w = jax.vjp(jax.jit(warp), dr)
    g = jnp.asarray(rng.normal(size=ref.shape))
    (gr,) = vjp_ref(g)
    (gw,) = vjp_w(g.astype(jnp.float32))
    # measured on an L40S: forward 2.5e-7, VJP 4.4e-7 relative to the max
    assert float(jnp.abs(out - ref).max() / jnp.abs(ref).max()) < 5e-6
    assert float(jnp.abs(gw - gr).max() / jnp.abs(gr).max()) < 1e-5
