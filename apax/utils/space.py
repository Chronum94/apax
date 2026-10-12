"""Small geometry helpers. Box convention: lattice vectors are the columns of
`box` (the transpose of an ASE cell), so real positions are `box @ frac`."""

import jax.numpy as jnp
from jax import Array, custom_jvp


def raw_transform(box: Array, R: Array) -> Array:
    """Apply `box` to every vector in `R` (`(..., 3)`).
    A scalar `box` scales, a matrix `box` maps each row `r` to `box @ r`.
    """
    box = jnp.asarray(box)
    if box.ndim == 0:
        return R * box
    return jnp.einsum("ij,...j->...i", box, R)


@custom_jvp
def transform(box: Array, R: Array) -> Array:
    """Like `raw_transform`, but differentiating through it leaves tangents
    of `R` unscaled: gradients taken w.r.t. fractional positions come back
    in real space (energy / length). The tangent of `box` is applied as usual.
    """
    return raw_transform(box, R)


@transform.defjvp
def _transform_jvp(primals, tangents):
    box, R = primals
    dbox, dR = tangents
    return transform(box, R), dR + transform(dbox, R)


def distance(dR: Array) -> Array:
    """Norm over the last axis, with a zero (finite-gradient) result for dR = 0."""
    dr_sq = jnp.sum(dR**2, axis=-1)
    nonzero = dr_sq > 0
    return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, dr_sq, 1.0)), 0.0)


def volume(box: Array) -> Array:
    """Signed volume of a `(3, 3)` box."""
    return jnp.linalg.det(box)
