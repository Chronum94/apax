import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apax.utils import space

BOX = jnp.array([[5.0, 0.7, -0.3], [0.0, 6.0, 0.4], [0.0, 0.0, 5.5]])


def pair_energy(transform_fn, distance_fn, R_frac, box):
    R = transform_fn(box, R_frac)
    dR = R[:, None, :] - R[None, :, :]
    r = distance_fn(dR)
    return jnp.sum(jnp.where(r > 0, jnp.exp(-r), 0.0))


@pytest.fixture
def R_frac():
    return jax.random.uniform(jax.random.PRNGKey(0), (6, 3))


def test_transform_gradients_are_real_space(R_frac):
    """d/dR_frac of E(transform(box, R_frac)) is the real-space gradient."""
    grad_frac = jax.grad(pair_energy, argnums=2)(
        space.transform, space.distance, R_frac, BOX
    )
    R_real = space.raw_transform(BOX, R_frac)
    grad_real = jax.grad(lambda R: pair_energy(lambda b, x: x, space.distance, R, BOX))(
        R_real
    )
    np.testing.assert_allclose(grad_frac, grad_real, rtol=1e-12)


def test_distance_at_zero_is_finite():
    value, grad = jax.value_and_grad(lambda x: space.distance(x))(jnp.zeros(3))
    assert value == 0.0
    assert jnp.all(grad == 0.0)


def test_raw_transform_scalar_and_matrix(R_frac):
    np.testing.assert_allclose(space.raw_transform(2.0, R_frac), 2.0 * R_frac)
    np.testing.assert_allclose(space.raw_transform(BOX, R_frac), R_frac @ BOX.T)


def test_matches_jax_md(R_frac):
    jax_md_space = pytest.importorskip("jax_md.space")

    def energy(module):
        return lambda R, box: pair_energy(module.transform, module.distance, R, box)

    ref = jax.value_and_grad(energy(jax_md_space), argnums=(0, 1))(R_frac, BOX)
    new = jax.value_and_grad(energy(space), argnums=(0, 1))(R_frac, BOX)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(new)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)

    np.testing.assert_allclose(
        space.raw_transform(1.01, R_frac), jax_md_space.raw_transform(1.01, R_frac)
    )
