import jax
import jax.numpy as jnp
import numpy as np

from gmnn_jax.layers.descriptor.gaussian_moment_descriptor import (
    GaussianMomentDescriptorFlax,
)


def test_gaussian_moment_descriptor():
    R = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    Z = jnp.array([8, 1, 1])
    neighbor = jnp.array([[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]])
    box = np.array([0.0, 0.0, 0.0])

    descriptor = GaussianMomentDescriptorFlax()

    key = jax.random.PRNGKey(0)
    params = descriptor.init(key, R, Z, neighbor, box)
    result = descriptor.apply(params, R, Z, neighbor, box)
    result_jit = jax.jit(descriptor.apply)(params, R, Z, neighbor, box)

    assert result.shape == (3, 360)
    assert result_jit.shape == (3, 360)
