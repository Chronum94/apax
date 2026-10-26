"""Tabulated radial functions for inference.

With trained parameters, each radial channel of an element pair (centre, neighbour) is a
fixed linear combination of the radial's basis functions. `tabulate_radial` recovers one
(n_basis, n_radial) coefficient matrix per element pair from samples of the radial's
public `apply` (least squares, residual-checked) and returns a `TabulatedRadialFunction`
with the same call signature, to be injected through `ModelBuilder(..., radial_fn=...)`.
The result is exact and removes the factorized radial's rank intermediate.
"""

from typing import Any, Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from apax.layers.descriptor.basis_functions import IdentityRadialTransform, cosine_cutoff


class TabulatedRadialFunction(nn.Module):
    """Radial function of element pairs from per-pair coefficient matrices on basis_fn,
    with the call signature of the exact radial functions.

    species_map: atomic number -> compact element index;
    pair type = index(centre) * n_elements + index(neighbour).
    table: (n_types, n_basis, n_radial) coefficients.
    """

    r_max: float
    species_map: np.ndarray
    table: np.ndarray
    basis_fn: Any
    dtype: Any = jnp.float32

    @property
    def _n_radial(self):
        return self.table.shape[-1]

    def __call__(self, dr, Z_i, Z_j):
        # Z_i: neighbour, Z_j: centre (as in RadialFunction / FactorizedRadialFunction)
        n_el = int(round(np.sqrt(self.table.shape[0])))
        species_map = jnp.asarray(self.species_map)
        pair_type = species_map[Z_j] * n_el + species_map[Z_i]
        dr = dr.astype(self.dtype)
        coef = jnp.asarray(self.table, self.dtype)[pair_type]
        radial = jnp.einsum("pb,pbr->pr", self.basis_fn(dr), coef)
        return radial * cosine_cutoff(dr, self.r_max)[:, None]


def tabulate_radial(
    radial_fn: nn.Module,
    radial_params,
    elements: Sequence[int],
    r_min: float = 0.5,
    dtype: Any = jnp.float32,
) -> TabulatedRadialFunction:
    """Fit the per-pair coefficients of a trained radial for every (centre, neighbour)
    pair of `elements`.

    radial_fn / radial_params: the exact module and its parameter subtree; build
    radial_fn in float64 for sampling.
    r_min: start of the sampled range in Angstrom.
    dtype: dtype of the table and of its evaluation.
    """
    transform = getattr(radial_fn, "radial_transform", IdentityRadialTransform())
    if not isinstance(transform, IdentityRadialTransform):
        raise ValueError("tabulation needs a radial that is linear in its basis")
    elements = sorted({int(z) for z in elements})
    if not elements or min(elements) < 1:
        raise ValueError(f"elements must be atomic numbers >= 1, got {elements}")
    species_map = np.zeros(119, dtype=np.int32)
    species_map[elements] = np.arange(len(elements))
    r_max = float(radial_fn.basis_fn.r_max)

    # sample short of r_max, where the cutoff is ~0
    d = jnp.linspace(r_min, 0.95 * r_max, 4 * radial_fn.basis_fn.n_basis)
    basis = np.asarray(radial_fn.basis_fn.apply({}, d), np.float64)
    cut = np.asarray(cosine_cutoff(d, r_max), np.float64)[:, None]
    tables = []
    for zc in elements:
        for zn in elements:
            z_nbr, z_centre = jnp.full(d.shape, zn), jnp.full(d.shape, zc)
            value = radial_fn.apply({"params": radial_params}, d, z_nbr, z_centre)
            value = np.asarray(value, np.float64)
            coef, *_ = np.linalg.lstsq(basis, value / cut, rcond=None)
            resid = np.abs(basis @ coef * cut - value).max()
            resid /= np.abs(value).max() + 1e-30
            if resid > 1e-6:
                raise ValueError(
                    f"radial is not linear in its basis for pair ({zc}, {zn}): "
                    f"residual {resid:.1e}"
                )
            tables.append(coef)
    return TabulatedRadialFunction(
        r_max=r_max,
        species_map=species_map,
        table=np.stack(tables).astype(np.dtype(jnp.dtype(dtype))),
        basis_fn=radial_fn.basis_fn.clone(dtype=dtype),
        dtype=dtype,
    )
