import logging
from dataclasses import field
from typing import Any, Callable, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax_md import partition, space

from apax.layers.adaptive_cutoff import AdaptiveCutoff
from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.distances import make_distance_fn
from apax.layers.empirical import EmpiricalEnergyTerm
from apax.layers.masking import mask_by_atom, mask_hessian
from apax.layers.properties import stress_times_vol
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.utils.math import fp64_sum
from apax.utils.transform import make_energy_only_model

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

log = logging.getLogger(__name__)


def apply_representation(representation, adaptive, dr_vec, Z, idx):
    """Representation on the raw edges, or on the adaptively selected and packed
    ones. Returns ``(features, dr_vec, idx)`` with the edges actually used."""
    if adaptive is None:
        return representation(dr_vec, Z, idx), dr_vec, idx
    dr_vec, idx, cutoffs, _ = adaptive(dr_vec, idx, Z.shape[0])
    # packed idx is traced; Z may be a closed-over numpy array
    Z = jnp.asarray(Z)
    return representation(dr_vec, Z, idx, cutoffs=cutoffs), dr_vec, idx


class FeatureModel(nn.Module):
    """Model wraps some submodel (e.g. a descriptor) to supply distance computation."""

    representation: nn.Module = GaussianMomentDescriptor()
    readout: nn.Module = AtomisticReadout()
    should_average: bool = False
    init_box: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    inference_disp_fn: Any = None
    mask_atoms: bool = True
    adaptive: Optional[AdaptiveCutoff] = None

    def setup(self):
        self.compute_distances = make_distance_fn(self.init_box, self.inference_disp_fn)

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

        features, *_ = apply_representation(
            self.representation, self.adaptive, dr_vec, Z, idx
        )

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
    adaptive: Optional[AdaptiveCutoff] = None

    def setup(self):
        self.compute_distances = make_distance_fn(self.init_box, self.inference_disp_fn)

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

        # Heads and corrections that use absolute positions expect Cartesian ones;
        # jax-md inference on periodic systems works in fractional coordinates.
        R_cart = R
        if self.inference_disp_fn is not None and not np.all(self.init_box < 1e-6):
            R_cart = space.transform(box, R)

        # Empirical corrections act on the raw neighbors, independent of selection
        dr_vec_raw, idx_raw = dr_vec, idx
        # Model Core
        # shape Natoms
        # shape shallow ens: Natoms x Nensemble
        g, dr_vec, idx = apply_representation(
            self.representation, self.adaptive, dr_vec, Z, idx
        )
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
            result = property_head(g, R_cart, dr_vec, Z, idx, box)
            properties.update(result)

        # Corrections
        for correction in self.corrections:
            energy_correction = correction(
                R_cart, dr_vec_raw, Z, idx_raw, box, properties
            )
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


def make_mean_energy_fn(energy_fn):
    def mean_energy_fn(
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
        perturbation=None,
    ):
        e_ens, _ = energy_fn(R, Z, neighbor, box, offsets, perturbation)
        E_mean = jnp.mean(e_ens)
        return E_mean

    return mean_energy_fn


def make_member_chunk_jac(energy_model, start, end):
    def energy_chunk_fn(R, Z, neighbor, box, offsets):
        Ei = energy_model(R, Z, neighbor, box, offsets)[start:end]
        return Ei

    grad_i_fn = jax.jacrev(energy_chunk_fn)
    return grad_i_fn


class ShallowEnsembleModel(nn.Module):
    """Transforms an EnergyModel into one that also predicts derivatives the total energy.
    Can calculate forces and stress tensors.
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
        energy_ens, properties = self.energy_model(R, Z, neighbor, box, offsets)
        # The two functions below drop the calculation of properties
        mean_energy_fn = make_mean_energy_fn(self.energy_model)
        energy_fn = make_energy_only_model(self.energy_model)

        n_ens = energy_ens.shape[0]
        divisor = 1 / (n_ens - 1)
        energy_mean = jnp.mean(energy_ens)
        energy_variance = divisor * fp64_sum((energy_ens - energy_mean) ** 2)

        prediction = {
            "energy": energy_mean,
            "energy_ensemble": energy_ens,
            "energy_uncertainty": jnp.sqrt(energy_variance),
        }
        prediction.update(properties)

        if self.force_variance:
            if not self.chunk_size:
                forces_ens = -jax.jacrev(energy_fn)(R, Z, neighbor, box, offsets)
            else:
                with jax.ensure_compile_time_eval():
                    if not n_ens % self.chunk_size == 0:
                        m = "the chunksize needs to be a factor of the number of ensemble members"
                        raise ValueError(m)

                forces_ens = []
                start = 0
                for _ in range(n_ens // self.chunk_size):
                    end = start + self.chunk_size
                    jac_i_fn = make_member_chunk_jac(energy_fn, start, end)
                    force_i = -jac_i_fn(R, Z, neighbor, box, offsets)
                    forces_ens.append(force_i)
                    start = end

                n_atoms = R.shape[0]
                forces_ens = jnp.array(forces_ens)
                forces_ens = np.reshape(forces_ens, (n_ens, n_atoms, 3))

            forces_mean = jnp.mean(forces_ens, axis=0)
            forces_variance = divisor * fp64_sum((forces_ens - forces_mean) ** 2, axis=0)

            prediction["forces"] = forces_mean
            prediction["forces_uncertainty"] = jnp.sqrt(forces_variance)

            forces_ens = jnp.transpose(forces_ens, (1, 2, 0))
            prediction["forces_ensemble"] = forces_ens  # n_atoms x 3 x n_members

        else:
            forces_mean = -jax.grad(mean_energy_fn)(R, Z, neighbor, box, offsets)
            prediction["forces"] = forces_mean

        if self.calc_stress:
            stress = stress_times_vol(
                mean_energy_fn, R, box, Z=Z, neighbor=neighbor, offsets=offsets
            )
            prediction["stress"] = stress

        if self.calc_hessian:
            hessian = jax.hessian(mean_energy_fn)(R, Z, neighbor, box, offsets)
            prediction["hessian"] = mask_hessian(hessian, Z)

        return prediction
