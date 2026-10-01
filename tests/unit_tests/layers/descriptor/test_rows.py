import jax
import jax.numpy as jnp
import numpy as np

from apax.layers.descriptor.basis_functions import GaussianBasis, RadialFunction
from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor


def test_rows_match_sparse():
    rng = np.random.default_rng(0)
    N, P, pad = 20, 300, 40
    idx = rng.integers(0, N, (2, P))
    idx = idx[:, idx[0] != idx[1]]
    dr = rng.uniform(-3, 3, (idx.shape[1], 3))
    idx = jnp.asarray(np.pad(idx, ((0, 0), (0, pad))))  # (0,0) padding pairs
    dr = jnp.asarray(np.pad(dr, ((0, pad), (0, 0))))
    Z = jnp.asarray(rng.integers(1, 10, N))
    K = int(np.bincount(np.asarray(idx[1, :-pad])).max())
    rf = RadialFunction(
        n_radial=4, basis_fn=GaussianBasis(n_basis=5, r_max=6.0), dtype="fp64"
    )
    sparse = GaussianMomentDescriptor(radial_fn=rf, dtype="fp64")
    rows = GaussianMomentDescriptor(radial_fn=rf, dtype="fp64", nbr_rows=K)
    params = sparse.init(jax.random.PRNGKey(0), dr, Z, idx)

    def run(m):
        return jax.value_and_grad(lambda d: (m.apply(params, d, Z, idx) ** 2).sum())(dr)

    (e0, g0), (e1, g1) = run(sparse), run(rows)
    np.testing.assert_allclose(e1, e0, rtol=1e-12)
    np.testing.assert_allclose(g1, g0, rtol=1e-10, atol=1e-12)
