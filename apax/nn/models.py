import logging
from dataclasses import field
from typing import Any, Callable, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_md import partition

from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.distances import make_distance_fn
from apax.layers.empirical import EmpiricalEnergyTerm
from apax.layers.masking import mask_by_atom, mask_hessian
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.utils.math import fp64_sum, safe_sqrt
from apax.utils.transform import make_energy_only_model

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

log = logging.getLogger(__name__)


class FeatureModel(nn.Module):
    """Model wraps some submodel (e.g. a descriptor) to supply distance computation."""

    representation: nn.Module = GaussianMomentDescriptor()
    readout: nn.Module = AtomisticReadout()
    should_average: bool = False
    init_box: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    inference_disp_fn: Any = None
    mask_atoms: bool = True
    geometry_dtype: Any = jnp.float64

    def setup(self):
        self.compute_distances = make_distance_fn(
            self.init_box, self.inference_disp_fn, self.geometry_dtype
        )

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
        perturbation=None,
    ):
        dr_vec, idx = self.compute_distances(
            R,
            neighbor,
            box,
            offsets,
            perturbation,
        )

        features = self.representation(dr_vec, Z, idx)

        if self.readout:
            features = jax.vmap(self.readout)(features)

        if self.mask_atoms:
            features = mask_by_atom(features, Z)
        if self.should_average:
            features = jnp.mean(features, axis=0)
        return features


class EnergyModel(nn.Module):
    """Model which post processes the output of an atomistic model and
    adds empirical energy terms.
    """

    representation: nn.Module = GaussianMomentDescriptor()
    readout: nn.Module = AtomisticReadout()
    scale_shift: nn.Module = PerElementScaleShift()
    property_heads: list[nn.Module] = field(default_factory=lambda: [])
    corrections: list[EmpiricalEnergyTerm] = field(default_factory=lambda: [])
    init_box: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    mask_atoms: bool = True
    inference_disp_fn: Any = None
    geometry_dtype: Any = jnp.float64

    def setup(self):
        self.compute_distances = make_distance_fn(
            self.init_box, self.inference_disp_fn, self.geometry_dtype
        )

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
        perturbation=None,
    ):
        dr_vec, idx = self.compute_distances(
            R,
            neighbor,
            box,
            offsets,
            perturbation,
        )

        # Model Core
        # shape Natoms
        # shape shallow ens: Natoms x Nensemble
        g = self.representation(dr_vec, Z, idx)
        h = jax.vmap(self.readout)(g)
        E_i = self.scale_shift(h, Z)

        if self.mask_atoms:
            E_i = mask_by_atom(E_i, Z)

        # check for shallow ensemble
        is_shallow_ensemble = E_i.shape[1] > 1
        if is_shallow_ensemble:  # is this necessary or is using sum with axis=0 enough?
            total_energies_ensemble = fp64_sum(E_i, axis=0)
            # shape Nensemble
            energy = total_energies_ensemble
        else:
            # shape ()
            energy = fp64_sum(E_i)

        properties = {}
        for property_head in self.property_heads:
            result = property_head(g, R, dr_vec, Z, idx, box)
            properties.update(result)

        # Corrections
        for correction in self.corrections:
            energy_correction = correction(R, dr_vec, Z, idx, box, properties)
            energy = energy + energy_correction

        return energy, properties


class EnergyDerivativeModel(nn.Module):
    """Transforms an EnergyModel into one that also predicts derivatives the total energy.
    Can calculate forces and stress tensors.
    """

    # Alternatively, should this be a function transformation?
    energy_model: EnergyModel = EnergyModel()
    calc_stress: bool = False
    calc_hessian: bool = False

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
    ):
        energy_only_model = make_energy_only_model(self.energy_model)
        if self.calc_stress:
            # forces and stress from one backward pass: differentiate w.r.t. the
            # positions and a deformation gradient F evaluated at the identity.
            # dE/dF at F = I is the stress times the cell volume.
            def energy_fn(R, F):
                return self.energy_model(R, Z, neighbor, box, offsets, F)

            ef_function = jax.value_and_grad(energy_fn, argnums=(0, 1), has_aux=True)
            identity = jnp.eye(R.shape[1], dtype=R.dtype)
            (energy, properties), (neg_forces, dE_dF) = ef_function(R, identity)
            prediction = {"energy": energy, "forces": -neg_forces}
            prediction.update(properties)
            prediction["stress"] = 0.5 * (dE_dF + dE_dF.T)
        else:
            ef_function = jax.value_and_grad(self.energy_model, has_aux=True)
            (energy, properties), neg_forces = ef_function(R, Z, neighbor, box, offsets)
            prediction = {"energy": energy, "forces": -neg_forces}
            prediction.update(properties)

        if self.calc_hessian:
            hessian = jax.hessian(energy_only_model)(R, Z, neighbor, box, offsets)
            prediction["hessian"] = mask_hessian(hessian, Z)

        return prediction


def _member_std(x):
    """Standard deviation over the leading (member) axis, with Bessel's correction."""
    n_ens = x.shape[0]
    mean = jnp.mean(x, axis=0)
    # safe_sqrt: members that agree exactly (padded atoms, components fixed by
    # symmetry) would otherwise give an infinite derivative and NaN gradients
    return safe_sqrt(fp64_sum((x - mean) ** 2, axis=0) / (n_ens - 1))


def _symmetrize(dE_dF):
    return 0.5 * (dE_dF + jnp.swapaxes(dE_dF, -1, -2))


class ShallowEnsembleModel(nn.Module):
    """Transforms an EnergyModel with a shallow-ensemble readout into one that
    predicts the ensemble mean and spread of energies, forces and stresses.

    With `force_variance`, the per-member forces (and stresses) come from one forward
    pass and one VJP per member, vmapped over the members (or `lax.map`ped in batches
    of `chunk_size`). Without it, only the gradient of the mean energy is computed.
    """

    energy_model: EnergyModel = EnergyModel()
    calc_stress: bool = False
    calc_hessian: bool = False
    force_variance: bool = True
    chunk_size: Optional[int] = None

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
    ):
        def energy_fn(R, F):
            return self.energy_model(R, Z, neighbor, box, offsets, F)

        identity = jnp.eye(R.shape[1], dtype=R.dtype)

        if self.force_variance:
            energy_ens, vjp_fn, properties = jax.vjp(energy_fn, R, identity, has_aux=True)
            n_ens = energy_ens.shape[0]
            cotangents = jnp.eye(n_ens, dtype=energy_ens.dtype)
            if self.chunk_size:
                # checkpoint: differentiating through the map (training) would otherwise
                # keep every chunk's intermediates, i.e. the memory of all members at once
                dE_dR, dE_dF = jax.lax.map(
                    jax.checkpoint(vjp_fn), cotangents, batch_size=self.chunk_size
                )
            else:
                dE_dR, dE_dF = jax.vmap(vjp_fn)(cotangents)
            forces_ens = -dE_dR  # n_members x n_atoms x 3

            forces = jnp.mean(forces_ens, axis=0)
            stress_ens = _symmetrize(dE_dF)
        else:

            def mean_energy(R, F):
                energy_ens, properties = energy_fn(R, F)
                return jnp.mean(energy_ens), (energy_ens, properties)

            grad_fn = jax.grad(mean_energy, argnums=(0, 1), has_aux=True)
            (dE_dR, dE_dF), (energy_ens, properties) = grad_fn(R, identity)
            forces = -dE_dR

        prediction = {
            "energy": jnp.mean(energy_ens),
            "energy_ensemble": energy_ens,
            "energy_uncertainty": _member_std(energy_ens),
            "forces": forces,
        }
        prediction.update(properties)

        if self.force_variance:
            prediction["forces_uncertainty"] = _member_std(forces_ens)
            # n_atoms x 3 x n_members
            prediction["forces_ensemble"] = jnp.transpose(forces_ens, (1, 2, 0))

        if self.calc_stress:
            if self.force_variance:
                prediction["stress"] = jnp.mean(stress_ens, axis=0)
                prediction["stress_uncertainty"] = _member_std(stress_ens)
                prediction["stress_ensemble"] = jnp.transpose(stress_ens, (1, 2, 0))
            else:
                prediction["stress"] = _symmetrize(dE_dF)

        if self.calc_hessian:
            hessian = jax.hessian(lambda R: jnp.mean(energy_fn(R, identity)[0]))(R)
            prediction["hessian"] = mask_hessian(hessian, Z)

        return prediction
