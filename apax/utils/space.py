"""Small geometry helpers. Box convention: lattice vectors are the columns of
`box` (the transpose of an ASE cell), so real positions are `box @ frac`."""

import jax.numpy as jnp
import numpy as np
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


def perpendicular_widths(box) -> np.ndarray:
    """Distances between opposite faces of the cell spanned by the columns of `box`."""
    box = np.asarray(box)
    a, b, c = box.T
    face_areas = np.linalg.norm([np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=1)
    return abs(np.linalg.det(box)) / face_areas


def _apply3(box: Array, R: Array) -> Array:
    """box @ r for every r in R (..., 3), written as explicit multiply-adds. Same values
    and dtype promotion as `raw_transform` / jax-md, but XLA cannot lower it as a dot: on
    GPUs a dot of millions of 3-vectors becomes a separate (slow) GEMM fusion, while this
    fuses with the surrounding elementwise work. Scalar / diagonal boxes use the einsum."""
    box = jnp.asarray(box)
    if box.ndim != 2 or box.shape != (3, 3) or R.shape[-1] != 3:
        return raw_transform(box, R) if box.ndim != 1 else R * box
    return R[..., 0:1] * box[:, 0] + R[..., 1:2] * box[:, 1] + R[..., 2:3] * box[:, 2]


@custom_jvp
def transform_elementwise(box: Array, R: Array) -> Array:
    """`transform` (and jax-md's) with the elementwise box product: tangents of R pass
    through unscaled, so gradients w.r.t. fractional positions come back in real space."""
    return _apply3(box, R)


@transform_elementwise.defjvp
def _transform_elementwise_jvp(primals, tangents):
    box, R = primals
    dbox, dR = tangents
    return transform_elementwise(box, R), dR + transform_elementwise(dbox, R)


def periodic_general_elementwise(box: Array, wrapped: bool = True):
    """jax_md.space.periodic_general(box, fractional_coordinates=True, wrapped) with an
    elementwise displacement (see `_apply3`): identical results, no GEMM fusions in
    neighbour-list builds (millions of candidate pairs) or pair-distance evaluation.
    The shift function is jax-md's."""
    from jax_md import space as jspace

    box = jnp.asarray(box)
    _, shift_fn = jspace.periodic_general(
        box, fractional_coordinates=True, wrapped=wrapped
    )

    def displacement_fn(Ra, Rb, perturbation=None, **kwargs):
        dR = (
            jnp.mod(Ra - Rb + 0.5, 1.0) - 0.5
        )  # jax_md.space.periodic_displacement(1.0, .)
        # Default (captured) box follows the positions' dtype: neighbour lists are built in
        # fp32 (nbr_f32) and must not be promoted to fp64 by an fp64 system box (fp64 is
        # ~1/64 rate on many GPUs). An explicitly passed box keeps jax-md's promotion.
        _box = kwargs.get("box", box.astype(dR.dtype))
        if "new_box" in kwargs:
            _box = kwargs["new_box"]
        dR = transform_elementwise(_box, dR)
        if perturbation is not None:
            dR = _apply3(perturbation, dR)
        return dR

    # lets compute_distances use the fused Warp pair-vector kernels (neighbor_grad.pair_vectors)
    displacement_fn.fractional_minimum_image = True
    return displacement_fn, shift_fn
