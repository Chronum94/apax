import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax_md import partition, space


def canonicalize_neighbors(neighbor):
    return neighbor.idx if isinstance(neighbor, partition.NeighborList) else neighbor


def get_disp_fn(displacement):
    def disp_fn(ri, rj, perturbation, box):
        return displacement(ri, rj, perturbation, box=box)

    return disp_fn


def make_distance_fn(init_box, inference_disp_fn=None):
    """Edge vectors ``r_ij = R_j - R_i (+ periodic shift)`` for a neighbor list.

    Without ``inference_disp_fn`` (training, batch evaluation, vesin-based
    inference), positions are Cartesian and ``offsets`` are the Cartesian image
    shifts ``S @ cell`` (zero for gas phase), so periodic and gas-phase structures
    share one code path and can be mixed in a dataset.

    With an ``inference_disp_fn`` (jax-md neighbor lists), the jax-md displacement
    function is used, with fractional positions for periodic systems.

    ``perturbation`` is the deformation gradient ``F = 1 + eps`` used for stress:
    every edge vector, including its periodic shift, is mapped to ``F r_ij``
    (jax_md convention).
    """

    if inference_disp_fn is None:

        def compute_distances(R, neighbor, box, offsets, perturbation=None):
            idx = canonicalize_neighbors(neighbor)
            R = R.astype(jnp.float64)
            dr_vec = R[idx[1]] - R[idx[0]]
            # Gas-phase callers may pass a zero placeholder that is not per-edge.
            if jnp.ndim(offsets) == 2 and offsets.shape[0] == dr_vec.shape[0]:
                dr_vec = dr_vec + offsets
            if perturbation is not None:
                dr_vec = space.raw_transform(perturbation, dr_vec)
            return dr_vec, idx

        return compute_distances

    if np.all(init_box < 1e-6):
        displacement = space.map_bond(space.free()[0])
    else:
        displacement = vmap(get_disp_fn(inference_disp_fn), (0, 0, None, None), 0)

    def compute_distances(R, neighbor, box, offsets, perturbation=None):
        idx = canonicalize_neighbors(neighbor)
        R = R.astype(jnp.float64)
        Ri = R[idx[0]]
        Rj = R[idx[1]]

        if np.all(init_box < 1e-6):
            dr_vec = displacement(Rj, Ri)
        else:
            dr_vec = displacement(Rj, Ri, None, box)
            dr_vec += offsets
            if perturbation is not None:
                # homogeneous deformation of the full pair vector, image shift
                # included (jax_md convention: F replaces dR)
                dr_vec = space.raw_transform(perturbation, dr_vec)
        return dr_vec, idx

    return compute_distances
