from dataclasses import field
from typing import Any

import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
from ase import data
from jax import vmap

from apax.layers.masking import mask_by_neighbor
from apax.layers.readout import AtomisticReadout
from apax.utils.jax_md_reduced import space
from apax.utils.math import fp64_sum


def stress_times_vol(energy_fn, position: Array, box, **kwargs) -> Array:
    """Computes the internal stress of a system multiplied with the box volume.
    For training purposes.

    Parameters
    ----------
    energy_fn:
        A function that computes the energy of the system. This
        function must take as an argument `perturbation` which perturbs the
        box shape. Any energy function constructed using `smap` or in `energy.py`
        with a standard space will satisfy this property.
    position:
        An array of particle positions.
    box:
        A box specifying the shape of the simulation volume. Used to infer the
        volume of the unit cell.

    Returns
    -------
    Array
        A float specifying the stress of the system.
    """
    dim = position.shape[1]
    zero = jnp.zeros((dim, dim), position.dtype)
    zero = 0.5 * (zero + zero.T)
    identity = jnp.eye(dim, dtype=position.dtype)

    def U(eps):
        return energy_fn(position, box=box, perturbation=(identity + eps), **kwargs)

    dUdV = jax.grad(U)
    return dUdV(zero)




class PropertyHead(nn.Module):

    pname: str
    readout: nn.Module = AtomisticReadout()
    aggregation: str = "none"
    mode: str = "l0"
    apply_mask: bool = True


    def setup(self):

        n_species = 119
        scale_init = nn.initializers.constant(1.0)
        self.scale = self.param(
            "scale_per_element", scale_init, (n_species, 1), jnp.float64
        )

    def __call__(self, g, R, dr_vec, Z, idx, box):

        # TODO shallow ensemble

        h = jax.vmap(self.readout)(g)

        p_i = h * self.scale[Z]

        if self.mode == "l0":
            p_i = p_i
        elif self.mode == "l1":
            p_i = p_i * R
        elif self.mode == "symmetric_traceless_l2":
            r_hat = R / jnp.linalg.norm(R, axis=1)
            r_rt = jnp.einsum("ni, nj -> nij", r_hat, r_hat)
            I = jnp.eye(3)
            symmetrized = 3*r_rt - I
            p_i = p_i * symmetrized
        else:
            raise KeyError("unknown symmetry option")
        
        if self.aggregation == "none":
            result = p_i
        elif self.aggregation == "sum":
            result = fp64_sum(p_i)
        elif self.aggregation == "mean":
            natoms = R.shape[0]
            result = fp64_sum(p_i) / natoms
        else:
            raise KeyError("unknown aggregation")

        if self.apply_mask:
            pass


        return {self.pname: result}

