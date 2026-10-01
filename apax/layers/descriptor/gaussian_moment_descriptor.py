from typing import Any

import einops
import flax.linen as nn
import jax.numpy as jnp
from jax import vmap
from jax_md import space

from apax.layers.descriptor.basis_functions import RadialFunction
from apax.layers.descriptor.moments import geometric_moments
from apax.layers.descriptor.rows import (
    P2,
    P3,
    UNPACK2,
    UNPACK3,
    W2,
    W3,
    row_cells,
    row_moments,
    to_rows,
    unpack,
)
from apax.layers.descriptor.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.layers.masking import mask_by_neighbor
from apax.utils.convert import str_to_dtype


class GaussianMomentDescriptor(nn.Module):
    radial_fn: nn.Module = RadialFunction()
    n_contr: int = 8
    dtype: Any = jnp.float32
    apply_mask: bool = True
    nbr_rows: int = 0

    def setup(self):
        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn._n_radial

        self.distance = vmap(space.distance, 0, 0)

        self.triang_idxs_2d = tril_2d_indices(self.n_radial)
        self.triang_idxs_3d = tril_3d_indices(self.n_radial)

    def __call__(self, dr_vec, Z, neighbor_idxs):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        # Z shape n_atoms
        n_atoms = Z.shape[0]

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        dr = self.distance(dr_vec)

        # TODO: maybe try jnp where
        dr_repeated = einops.repeat(dr + 1e-5, "neighbors -> neighbors 1")
        # normalized distance vectors, shape neighbors x 3
        dn = dr_vec / dr_repeated

        radial_function = self.radial_fn(dr, Z_i, Z_j)
        if self.apply_mask:
            radial_function = mask_by_neighbor(radial_function, neighbor_idxs, dr_vec)

        if self.nbr_rows:
            # padded-row layout: K = self.nbr_rows neighbours per centre
            valid = (idx_i != idx_j) | (jnp.sum(dr_vec**2, axis=1) > 1e-12)
            cell = row_cells(idx_j, valid, n_atoms, self.nbr_rows)
            f = to_rows(radial_function, cell, n_atoms, self.nbr_rows)
            d = to_rows(dn, cell, n_atoms, self.nbr_rows)
            m0, m1, m2p, m3p = row_moments(f, d)
        else:
            m0, m1, m2, m3 = geometric_moments(radial_function, dn, idx_j, n_atoms)
            m2p = jnp.stack([m2[..., a, b] for a, b in P2], axis=-1)
            m3p = jnp.stack([m3[..., a, b, c] for a, b, c in P3], axis=-1)

        # contractions as broadcast-multiply + sum: einsum lowers to TF32 GEMMs
        M2 = unpack(m2p, UNPACK2).reshape(n_atoms, -1, 3, 3)
        M3 = unpack(m3p, UNPACK3).reshape(n_atoms, -1, 9, 3)
        w2, w3 = jnp.asarray(W2, dtype), jnp.asarray(W3, dtype)

        def pair(a, b):  # (N, r, c) x (N, s, c) -> (N, r, s), summed over c
            return (a[:, :, None] * b[:, None]).sum(-1)

        contr_0 = m0
        contr_1 = pair(m1, m1)
        contr_2 = pair(m2p, m2p * w2)
        contr_3 = pair(m3p, m3p * w3)
        # A_rs[j, k] = sum_i M2[r, i, j] M2[s, i, k]
        A = (M2[:, :, None, :, :, None] * M2[:, None, :, :, None, :]).sum(3)
        contr_4 = pair(A.reshape(n_atoms, -1, 9), M2.reshape(n_atoms, -1, 9))
        contr_4 = contr_4.reshape(n_atoms, *A.shape[1:3], -1)
        B = m1[:, :, None, :, None] * m1[:, None, :, None, :]
        contr_5 = pair(B.reshape(n_atoms, -1, 9), M2.reshape(n_atoms, -1, 9))
        contr_5 = contr_5.reshape(n_atoms, *B.shape[1:3], -1)
        # C_rs[k, l] = sum_ij M3[r, ij, k] M3[s, ij, l]
        C = (M3[:, :, None, :, :, None] * M3[:, None, :, :, None, :]).sum(3)
        contr_6 = pair(C.reshape(n_atoms, -1, 9), M2.reshape(n_atoms, -1, 9))
        contr_6 = contr_6.reshape(n_atoms, *C.shape[1:3], -1)
        # D_rs[k] = sum_ij M3[r, ij, k] M2[s, ij]
        D = (M3[:, :, None] * M2.reshape(n_atoms, 1, -1, 9, 1)).sum(3)
        contr_7 = pair(D.reshape(n_atoms, -1, 3), m1)

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

        gaussian_moments = [
            contr_0,
            contr_1,
            contr_2,
            contr_3,
            contr_4,
            contr_5,
            contr_6,
            contr_7,
        ]

        # gaussian_moments shape: n_atoms x n_features
        gaussian_moments = jnp.concatenate(gaussian_moments[: self.n_contr], axis=-1)
        assert gaussian_moments.dtype == dtype
        return gaussian_moments
