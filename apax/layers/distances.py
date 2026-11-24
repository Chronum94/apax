import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from apax.utils.space import raw_transform, transform


def canonicalize_neighbors(neighbor):
    return getattr(neighbor, "idx", neighbor)


def disp_fn(ri, rj, perturbation, box):
    # `perturbation` is accepted for signature compatibility but applied in
    # `compute_distances`, after the periodic-image offsets are added.
    dR = transform(box, ri - rj)
    return dR


def get_disp_fn(displacement):
    def disp_fn(ri, rj, perturbation, box):
        return displacement(ri, rj, perturbation, box=box)

    return disp_fn


def make_distance_fn(
    init_box, inference_disp_fn=None, geometry_dtype=jnp.float64, neighbor_grad="scatter"
):
    """Model which post processes the output of an atomistic model and
    adds empirical energy terms.
    """

    if np.all(init_box < 1e-6):
        # gas phase training and predicting
        def displacement(Ra, Rb):
            return Ra - Rb

    elif inference_disp_fn is None:
        # for training on periodic systems
        displacement = vmap(disp_fn, (0, 0, None, None), 0)
    else:
        mappable_displacement_fn = get_disp_fn(inference_disp_fn)
        displacement = vmap(mappable_displacement_fn, (0, 0, None, None), 0)

    # `geometry_dtype` is the precision of positions, displacements and their gradients.
    # f64 by default (training, ASE: stress consistent to finite differences); jax-md MD
    # uses f32, since every pair-sized gather/scatter (and its backward) in f64 is slow
    # on GPUs with low FP64 throughput and box-wrapped coordinates lose only ~1e-6 A.

    # MD fast path (E21): fractional minimum-image displacement in a constant box (tags set
    # by periodic_general_elementwise and md_setup) + "pull" gradient on a symmetric
    # centre-sorted list -> one fused Warp kernel each way for the pair vectors. The box is
    # traced inside the integrator even when constant, hence the setup-time tag.
    fused_pair_vectors = (
        neighbor_grad == "pull"
        and getattr(inference_disp_fn, "fractional_minimum_image", False)
        and getattr(inference_disp_fn, "constant_box", False)
        and jnp.dtype(geometry_dtype) == jnp.float32
    )

    def compute_distances(R, neighbor, box, offsets, perturbation=None):
        # Distances
        idx = canonicalize_neighbors(neighbor)
        idx_i, idx_j = idx[0], idx[1]

        # R shape n_atoms x 3
        R = R.astype(geometry_dtype)
        box = jnp.asarray(box, geometry_dtype)
        offsets = jnp.asarray(offsets, geometry_dtype)
        if perturbation is not None:
            perturbation = jnp.asarray(perturbation, geometry_dtype)
        if fused_pair_vectors and perturbation is None:
            # constant box only (no box cotangent): NVT / NVE MD forces; stress / NPT use the
            # perturbation or a non-constant box and take the general path below
            from apax.layers.neighbor_grad import pair_vectors

            return pair_vectors(R, idx, box) + offsets, idx
        if neighbor_grad == "pull":
            # jax-md Sparse lists only: one scatter-free backward for both gathers
            # (apax/layers/neighbor_grad.py)
            from apax.layers.neighbor_grad import gather_pair

            Rj, Ri = gather_pair(R, idx)
        else:
            Ri = R[idx_i]
            # idx_j (centre) is sorted in jax-md's sparse neighbour lists; flagging it lets
            # XLA accumulate the gather's backward scatter per run instead of atomics.
            Rj = jnp.take(R, idx_j, axis=0, mode="clip", indices_are_sorted=True)

        # dr_vec shape: neighbors x 3
        if np.all(init_box < 1e-6):
            # reverse conventnion to match TF
            # distance vector for gas phase training and predicting
            dr_vec = displacement(Rj, Ri)
        else:
            # distance vector for training on periodic systems
            dr_vec = displacement(Rj, Ri, None, box)
            dr_vec += offsets
            if perturbation is not None:
                # homogeneous deformation F = I + strain of the full pair vector,
                # periodic-image shift included (jax_md convention: F replaces dR)
                dr_vec = raw_transform(perturbation, dr_vec)
        return dr_vec, idx

    return compute_distances
