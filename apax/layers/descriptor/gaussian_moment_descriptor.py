from typing import Any, Optional

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import vmap

from apax.layers.descriptor.basis_functions import RadialFunction
from apax.layers.descriptor.moments import (
    _G2,
    _G3,
    geometric_moments,
    geometric_moments_packed,
    unpack_symmetric,
)
from apax.layers.descriptor.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.layers.masking import mask_by_neighbor
from apax.utils.convert import str_to_dtype
from apax.utils.space import distance


class GaussianMomentDescriptor(nn.Module):
    radial_fn: nn.Module = RadialFunction()
    n_contr: int = 8
    n_radial_tensor: Optional[int] = None
    packed_moments: bool = False
    drop_contractions: tuple = ()  # contraction blocks to leave out (XLA prunes their computation)
    dtype: Any = jnp.float32
    apply_mask: bool = True
    # "warp": fused NVIDIA Warp moment kernels (inference only; needs a
    # TabulatedRadialFunction with the Bessel basis and the identity transform)
    moment_kernel: Optional[str] = None
    # (a, b, r_max, rep_scale) of an NLHRepulsion fused into the Warp kernels; the descriptor
    # then returns (features, per-atom NLH energy) and the correction is not run separately
    fused_nlh: Any = None
    contract_kernel: str = "xla"  # "warp": fused contraction kernels (warp_contract.py)
    # return PackedMoments (with contract_kernel "warp"): the energy model contracts per atom block
    packed_only: bool = False

    def setup(self):
        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn._n_radial

        self.distance = vmap(distance, 0, 0)

        n_tensor = self.n_radial_tensor or self.n_radial
        if self.n_radial_tensor:
            # one learned map from the radial channels to the (fewer) channels of all l>0 moments,
            # applied per pair; initialised to keep the first n_radial_tensor channels
            self.radial_compression = nn.Dense(
                n_tensor,
                use_bias=False,
                kernel_init=lambda key, shape, dtype: jnp.eye(*shape, dtype=dtype),
                dtype=str_to_dtype(self.dtype),
                param_dtype=str_to_dtype(self.dtype),
                precision=jax.lax.Precision.HIGHEST,
            )

        self.triang_idxs_2d = tril_2d_indices(n_tensor)
        self.triang_idxs_3d = tril_3d_indices(n_tensor)

    def _warp_moments(self, dr_vec, Z, neighbor_idxs):
        from apax.layers.descriptor.basis_functions import (
            BesselBasis,
            IdentityRadialTransform,
        )
        from apax.layers.descriptor.radial_table import TabulatedRadialFunction
        from apax.layers.descriptor.warp_moments import (
            fold_compression,
            warp_packed_moments,
        )

        rf = self.radial_fn
        if not (
            isinstance(rf, TabulatedRadialFunction)
            and isinstance(rf.basis_fn, BesselBasis)
            and isinstance(rf.radial_transform, IdentityRadialTransform)
        ):
            raise ValueError(
                "moment_kernel='warp' needs a tabulated radial function with the Bessel "
                "basis and the identity radial transform"
            )
        if not self.apply_mask:
            raise ValueError("moment_kernel='warp' always masks padded pairs")
        T0 = jnp.asarray(rf.table, jnp.float32)
        K = T0.shape[-1]
        if self.n_radial_tensor:
            # the compression is linear without bias: its kernel is compression(I)
            W = self.radial_compression(jnp.eye(K, dtype=str_to_dtype(self.dtype)))
            T1 = fold_compression(T0, W.astype(jnp.float32))
        else:
            T1 = T0
        if self.contract_kernel == "warp":
            from apax.layers.descriptor.warp_contract import warp_contract

            if str_to_dtype(self.dtype) != jnp.float32:
                raise ValueError(
                    "contract_kernel='warp' computes in fp32 (descriptor_dtype fp32)"
                )
            res = warp_packed_moments(
                dr_vec,
                Z,
                neighbor_idxs,
                rf.species_map,
                T0,
                T1,
                rf.r_max,
                nlh=self.fused_nlh,
                packed=True,
            )
            if self.packed_only:
                from apax.layers.descriptor.warp_contract import PackedMoments

                feats = PackedMoments(
                    res[0],
                    int(K),
                    int(T1.shape[-1]),
                    int(self.n_contr),
                    tuple(self.drop_contractions),
                )
            else:
                feats = warp_contract(
                    res[0], K, T1.shape[-1], self.n_contr, tuple(self.drop_contractions)
                )
            return feats, (res[1] if self.fused_nlh is not None else None)
        res = warp_packed_moments(
            dr_vec, Z, neighbor_idxs, rf.species_map, T0, T1, rf.r_max, nlh=self.fused_nlh
        )
        m0, m1, p2, p3 = res[:4]
        e_nlh = res[4] if self.fused_nlh is not None else None
        dtype = str_to_dtype(self.dtype)
        return [
            m0.astype(dtype),
            m1.astype(dtype),
            unpack_symmetric(p2, _G2).astype(dtype),
            unpack_symmetric(p3, _G3).astype(dtype),
        ], e_nlh

    def __call__(self, dr_vec, Z, neighbor_idxs):
        if self.moment_kernel == "warp":
            moments, e_nlh = self._warp_moments(dr_vec, Z, neighbor_idxs)
            if self.contract_kernel == "warp":
                feats = moments  # already contracted
            else:
                feats = self._contract(moments, Z.shape[0], str_to_dtype(self.dtype))
            if e_nlh is not None:
                return feats, e_nlh
            return feats
        elif self.moment_kernel is not None:
            raise ValueError(f"unknown moment_kernel {self.moment_kernel!r}")
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

        moment_fn = geometric_moments_packed if self.packed_moments else geometric_moments
        if self.n_radial_tensor:
            moments = moment_fn(
                self.radial_compression(radial_function), dn, idx_j, n_atoms
            )
            moments[0] = jax.ops.segment_sum(
                radial_function, idx_j, n_atoms, indices_are_sorted=True
            )
        else:
            moments = moment_fn(radial_function, dn, idx_j, n_atoms)

        return self._contract(moments, n_atoms, dtype)

    def _contract(self, moments, n_atoms, dtype):
        contr_0 = moments[0]
        contr_1 = jnp.einsum("ari, asi -> ars", moments[1], moments[1])  # noqa: E501
        contr_2 = jnp.einsum("arij, asij -> ars", moments[2], moments[2])  # noqa: E501
        contr_3 = jnp.einsum("arijk, asijk -> ars", moments[3], moments[3])  # noqa: E501
        contr_4 = jnp.einsum(
            "arij, asik, atjk -> arst", moments[2], moments[2], moments[2]
        )  # noqa: E501
        contr_5 = jnp.einsum("ari, asj, atij -> arst", moments[1], moments[1], moments[2])  # noqa: E501
        contr_6 = jnp.einsum(
            "arijk, asijl, atkl -> arst", moments[3], moments[3], moments[2]
        )  # noqa: E501
        contr_7 = jnp.einsum(
            "arijk, asij, atk -> arst", moments[3], moments[2], moments[1]
        )  # noqa: E501

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
        gaussian_moments = jnp.concatenate(
            [
                g
                for k, g in enumerate(gaussian_moments[: self.n_contr])
                if k not in self.drop_contractions
            ],
            axis=-1,
        )
        assert gaussian_moments.dtype == dtype
        return gaussian_moments
