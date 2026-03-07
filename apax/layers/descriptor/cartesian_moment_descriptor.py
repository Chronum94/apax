from typing import Any

import jax.numpy as jnp
import flax.linen as nn
from jax import vmap

from apax.layers.descriptor.basis_functions import EmbeddedRadialFunction
from apax.layers.descriptor.moments import geometric_moments
from apax.layers.descriptor.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.layers.masking import mask_by_neighbor
from apax.utils.convert import str_to_dtype
from apax.utils.jax_md_reduced import space


class CartesianMomentDescriptor(nn.Module):
    """Rotationally-invariant descriptor based on Cartesian geometric moments.

    Functionally equivalent to ``GaussianMomentDescriptor`` but designed to
    work with displacement-vector-aware radial functions such as
    :class:`~apax.layers.descriptor.basis_functions.EmbeddedRadialFunction`,
    which accept ``(dr_vec, Z, neighbor_idxs)`` and handle species conditioning
    and cutoff internally.

    Scalar distances are computed internally only for the safe unit-vector
    normalisation used in the moment accumulation.

    Parameters
    ----------
    radial_fn : nn.Module
        Radial function with signature
        ``(dr_vec, Z, neighbor_idxs) → (n_pairs, n_radial)``.
        Must expose ``r_max`` and ``_n_radial`` attributes.
    n_contr : int
        Number of moment contractions to concatenate (1–8).
    dtype : dtype
        Descriptor compute dtype.
    apply_mask : bool
        Zero out contributions from padding pairs (where ``idx_i == idx_j``).
    """

    radial_fn: nn.Module = EmbeddedRadialFunction()
    n_contr: int = 8
    dtype: Any = jnp.float32
    apply_mask: bool = True

    def setup(self):
        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn._n_radial

        self.distance = vmap(space.distance, 0, 0)

        self.triang_idxs_2d = tril_2d_indices(self.n_radial)
        self.triang_idxs_3d = tril_3d_indices(self.n_radial)

    def __call__(self, dr_vec, Z, neighbor_idxs):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        n_atoms = Z.shape[0]

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]

        # Scalar distances for safe unit-vector computation only.
        dr = self.distance(dr_vec)  # (n_pairs,)

        # Safe norm: dn = 0 when dr ≈ 0 (no directional information).
        safe_dr = jnp.where(dr < 1e-8, jnp.ones_like(dr), dr)
        dn = jnp.where(
            dr[..., None] < 1e-8,
            jnp.zeros_like(dr_vec),
            dr_vec / safe_dr[..., None],
        )

        # Radial features: (n_pairs, n_radial).
        # Cutoff and species conditioning are handled inside radial_fn.
        radial_function = self.radial_fn(dr_vec, Z, neighbor_idxs)
        if self.apply_mask:
            radial_function = mask_by_neighbor(radial_function, neighbor_idxs)

        # Accumulate geometric moments at each center atom (idx_j convention).
        moments = geometric_moments(radial_function, dn, idx_j, n_atoms)

        contr_0 = moments[0]
        contr_1 = jnp.einsum("ari, asi -> ars", moments[1], moments[1])
        contr_2 = jnp.einsum("arij, asij -> ars", moments[2], moments[2])
        contr_3 = jnp.einsum("arijk, asijk -> ars", moments[3], moments[3])
        contr_4 = jnp.einsum("arij, asik, atjk -> arst", moments[2], moments[2], moments[2])
        contr_5 = jnp.einsum("ari, asj, atij -> arst", moments[1], moments[1], moments[2])
        contr_6 = jnp.einsum("arijk, asijl, atkl -> arst", moments[3], moments[3], moments[2])
        contr_7 = jnp.einsum("arijk, asij, atk -> arst", moments[3], moments[2], moments[1])

        tril_2_i, tril_2_j = self.triang_idxs_2d[:, 0], self.triang_idxs_2d[:, 1]
        tril_3_i, tril_3_j, tril_3_k = (
            self.triang_idxs_3d[:, 0],
            self.triang_idxs_3d[:, 1],
            self.triang_idxs_3d[:, 2],
        )

        contr_1 = contr_1[:, tril_2_i, tril_2_j]
        contr_2 = contr_2[:, tril_2_i, tril_2_j]
        contr_3 = contr_3[:, tril_2_i, tril_2_j]
        contr_4 = contr_4[:, tril_3_i, tril_3_j, tril_3_k]
        contr_5 = contr_5[:, tril_2_i, tril_2_j]
        contr_6 = contr_6[:, tril_2_i, tril_2_j]

        contr_5 = jnp.reshape(contr_5, [n_atoms, -1])
        contr_6 = jnp.reshape(contr_6, [n_atoms, -1])
        contr_7 = jnp.reshape(contr_7, [n_atoms, -1])

        contractions = [
            contr_0,
            contr_1,
            contr_2,
            contr_3,
            contr_4,
            contr_5,
            contr_6,
            contr_7,
        ]

        descriptor = jnp.concatenate(contractions[: self.n_contr], axis=-1)
        assert descriptor.dtype == dtype
        return descriptor
