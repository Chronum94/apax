"""Warp Gaussian-moment contractions (warp_contract.py) vs GaussianMomentDescriptor._contract.

Forward features and VJP on random packed moments, against an fp64 XLA reference, for the
full feature set and with n_contr / drop_contractions (the feature table must follow the
same order as _contract)."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("warp")

from apax.layers.descriptor.gaussian_moment_descriptor import (  # noqa: E402
    GaussianMomentDescriptor,
)
from apax.layers.descriptor.moments import _G2, _G3, unpack_symmetric  # noqa: E402
from apax.layers.descriptor.triangular_indices import (  # noqa: E402
    tril_2d_indices,
    tril_3d_indices,
)
from apax.layers.descriptor.warp_contract import (  # noqa: E402
    backward_chunks,
    chunk_length,
    feature_table,
    warp_contract,
)

requires_gpu = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="needs a CUDA GPU"
)

K, C = 12, 5


def _reference(packed, n_contr, drop):
    fake = types.SimpleNamespace(
        triang_idxs_2d=tril_2d_indices(C),
        triang_idxs_3d=tril_3d_indices(C),
        n_contr=n_contr,
        drop_contractions=drop,
    )
    moments = [
        packed[:, :K],
        packed[:, K : K + 3 * C].reshape(-1, C, 3),
        unpack_symmetric(packed[:, K + 3 * C : K + 9 * C].reshape(-1, C, 6), _G2),
        unpack_symmetric(packed[:, K + 9 * C :].reshape(-1, C, 10), _G3),
    ]
    return GaussianMomentDescriptor._contract(
        fake, moments, packed.shape[0], packed.dtype
    )


def _packed(n_atoms, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.concatenate(
        [np.full(K, 3.0), np.full(3 * C, 1.0), np.full(6 * C, 0.7), np.full(10 * C, 0.5)]
    )
    return rng.normal(size=(n_atoms, K + 19 * C)) * scale


def test_backward_work_list_covers_every_slot_once():
    table = tuple(map(tuple, feature_table(C).tolist()))
    chunks, entries = backward_chunks(table, 4)
    n_slots = {1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3}
    expected = sorted(
        (f, p) for f, row in enumerate(table) for p in range(n_slots[row[0]])
    )
    got = sorted(
        (int(entries[s + e]), int(pos))
        for (_, fam, pos, s, n, _) in chunks
        for e in range(n)
    )
    assert got == expected
    assert (
        chunk_length(3861) == 4
        and chunk_length(17875) == 16
        and chunk_length(10**6) == 32
    )


@requires_gpu
@pytest.mark.parametrize(
    "n_contr, drop", [(8, ()), (8, (3, 6)), (5, ())], ids=["all", "drop36", "ncontr5"]
)
# 257 and 5000 atoms: small and larger grids of the per-(family, position) kernels
@pytest.mark.parametrize("n_atoms", [257, 5000])
def test_warp_contract_matches_xla(n_atoms, n_contr, drop):
    p = _packed(n_atoms)
    ref, vjp_ref = jax.vjp(
        lambda x: _reference(x, n_contr, drop), jnp.asarray(p, jnp.float64)
    )
    out, vjp_w = jax.vjp(
        lambda x: warp_contract(x, K, C, n_contr, drop), jnp.asarray(p, jnp.float32)
    )
    assert out.shape == ref.shape
    g = jnp.asarray(np.random.default_rng(1).normal(size=ref.shape))
    (gr,) = vjp_ref(g)
    (gw,) = vjp_w(g.astype(jnp.float32))
    # fp32 kernels vs fp64 reference: ~1e-7 relative measured on an L40S
    assert float(jnp.abs(out - ref).max() / jnp.abs(ref).max()) < 1e-5
    assert float(jnp.abs(gw - gr).max() / jnp.abs(gr).max()) < 1e-5
