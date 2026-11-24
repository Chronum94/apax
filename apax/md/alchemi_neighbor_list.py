"""jax-md-compatible sparse neighbour lists built by NVIDIA ALCHEMI Toolkit-Ops
(`nvalchemi-toolkit-ops`, Warp cell lists with JAX bindings), as a drop-in for
`jax_md.partition.neighbor_list(..., format=Sparse, fractional_coordinates=True)` in MD.

Why: jax-md's cell-list build materialises every candidate pair (31M for 17.9k atoms, ~1,750
per atom for ~100 real neighbours) and prunes them with large scatters (~2.4 ms per rebuild on
an L40S, ~10 % of a step at a rebuild every ~8 steps). ALCHEMI's cell list never stores
candidates and returns a centre-sorted, padded, fixed-capacity COO list that is jit-compatible.

This module is only the thin adapter apax needs (we do not own the search kernels):
  * allocate: grid sizes (`estimate_cell_list_sizes`, not jittable) and capacities are fixed
    here; capacity = occupancy * capacity_multiplier + N * extra_capacity (jax-md semantics);
  * update: rebuild when any atom moved more than dr_threshold / 2 since the last build
    (jax-md's criterion), inside lax.cond;
  * output: jax-md `NeighborList` (idx[0] neighbour, idx[1] centre, padding = N; idx[2] the
    reverse-pair slot, see `_with_reverse`); overflow
    (capacity, per-atom max_neighbors, or ALCHEMI's validity flag) sets
    NEIGHBOR_LIST_OVERFLOW, so apax's reallocate-and-redo logic is unchanged.

Spatial sorting (`sort_atoms=True`, E14): at every rebuild the atoms are also renumbered in
Morton (Z-order) of their fractional positions and the list is built in that sorted index
space; the permutation travels with the list (`SortedNeighborList.perm` / `.inv`). The
permutation keeps the species pattern (each element is Morton-sorted into that element's own
index slots, so Z[perm] == Z): the model keeps its compile-time-constant Z, and XLA keeps
folding the atom masks and per-element tables (a runtime Z[perm] cost 0.36 ms/step at 17.9k
from changed fusions). The model call is wrapped by `sorted_frame`, which evaluates the model
on R[perm] and maps per-atom outputs back, so the integrator, trajectory, constraints and
biases keep the original atom order. Why: neighbour indices close to the centre index make the pair gathers
and per-atom atomics cache-friendly; MD diffusion scrambles any initial order within ~1 ns,
which cost 7-9 % (random vs spatially ordered input, 17.9k / 104k atoms, L40S).

Limits: periodic boxes with fractional coordinates, constant box (not NPT).
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import dataclasses, partition

from apax.utils.space import _apply3


@dataclasses.dataclass
class SortedNeighborList(partition.NeighborList):
    """jax-md NeighborList whose idx refers to atoms renumbered by perm:
    sorted atom s is original atom perm[s]; inv[perm] = arange(N)."""

    perm: Any = None
    inv: Any = None


def _spread_bits(v):
    """10-bit integers -> every third bit (for 3-D Morton codes)."""
    v = v & jnp.uint32(0x3FF)
    v = (v | (v << 16)) & jnp.uint32(0x030000FF)
    v = (v | (v << 8)) & jnp.uint32(0x0300F00F)
    v = (v | (v << 4)) & jnp.uint32(0x030C30C3)
    v = (v | (v << 2)) & jnp.uint32(0x09249249)
    return v


def morton_order(frac, species):
    """Species-preserving Morton permutation: perm[s] is the atom placed in slot s, with
    species[perm] == species. Atoms of each species, in Morton order of their wrapped
    fractional positions (1024 bins per axis), fill that species' slots in index order.
    species: static (N,) integer array."""
    species = np.asarray(species)
    g = jnp.clip(jnp.floor(jnp.mod(frac, 1.0) * 1024.0), 0, 1023).astype(jnp.uint32)
    key = _spread_bits(g[:, 0]) | (_spread_bits(g[:, 1]) << 1) | (_spread_bits(g[:, 2]) << 2)
    _, sp = np.unique(species, return_inverse=True)
    key = jnp.asarray(sp, jnp.uint64) << 30 | key.astype(jnp.uint64)  # species-major
    order = jnp.argsort(key, stable=True).astype(jnp.int32)
    slots = np.argsort(sp, kind="stable").astype(np.int32)  # each species' slots, in order
    N = frac.shape[0]
    perm = jnp.zeros((N,), jnp.int32).at[slots].set(order)
    inv = jnp.zeros((N,), jnp.int32).at[perm].set(jnp.arange(N, dtype=jnp.int32))
    return perm, inv


def sorted_frame(model, unpermute_outputs=False):
    """Wrap model(params, R, Z, neighbor, box, offsets, ...) to run in the sorted index space
    of a SortedNeighborList (identity for any other neighbour list). With unpermute_outputs,
    output leaves with a leading atom axis (N, ...) are mapped back to the original order
    (forces, per-atom uncertainties); scalars and the 3x3 stress are left alone."""

    @functools.wraps(model)
    def wrapped(params, R, Z, neighbor, box, offsets, *args, **kwargs):
        perm = getattr(neighbor, "perm", None)
        if perm is None:
            return model(params, R, Z, neighbor, box, offsets, *args, **kwargs)
        N = R.shape[0]
        # Z unchanged: the permutation preserves the species pattern (Z[perm] == Z)
        out = model(params, R[perm], Z, neighbor.idx, box, offsets, *args, **kwargs)
        if not unpermute_outputs:
            return out

        def back(x):
            if hasattr(x, "shape") and x.ndim >= 1 and x.shape[0] == N and x.shape != (3, 3):
                return x[neighbor.inv]
            return x

        return jax.tree_util.tree_map(back, out)

    return wrapped


def alchemi_neighbor_list(
    box, r_cutoff, dr_threshold, capacity_multiplier=1.25, sort_atoms=False, species=None
):
    """box: (3, 3) lattice vectors as columns (apax / jax-md convention); constant.
    sort_atoms: build in Morton-sorted index space (SortedNeighborList; wrap the model with
    `sorted_frame`); needs species, the (N,) atomic numbers."""
    if sort_atoms and species is None:
        raise ValueError("sort_atoms needs the atomic numbers (species)")
    try:
        import nvalchemiops.jax.neighbors as anb
    except ImportError as e:
        raise ImportError(
            "neighbor_builder: alchemi needs NVIDIA ALCHEMI Toolkit-Ops: "
            "pip install 'apax[alchemi]' (or nvalchemi-toolkit-ops), "
            "or use the default neighbor_builder: jax_md"
        ) from e

    cutoff = float(r_cutoff) + float(dr_threshold)
    threshold_sq = (0.5 * float(dr_threshold)) ** 2
    box32 = jnp.asarray(np.asarray(box), jnp.float32)
    cell = box32.T[None]  # ALCHEMI: lattice vectors as rows
    pbc = jnp.ones((3,), bool)
    static = {}  # grid sizes and per-atom capacity, fixed at allocation

    def _cart(frac):
        return _apply3(box32, jnp.asarray(frac, jnp.float32))

    def _search(frac, capacity):
        out = anb.cell_list(
            _cart(frac), cutoff, cell=cell, pbc=pbc,
            max_neighbors=static["max_neighbors"],
            max_total_cells=static["max_total_cells"],
            neighbor_search_radius=static["search_radius"],
            return_neighbor_list=True, coo_capacity=int(capacity),
        )
        coo, counts, valid = out[0], out[3], out[4]
        idx = jnp.stack([coo[1], coo[0]])  # jax-md: row 0 neighbour, row 1 centre
        overflow = (
            (jnp.sum(counts) > capacity)
            | (jnp.max(counts) > static["max_neighbors"])
            | jnp.logical_not(valid)
        )
        err = partition.PartitionError(jnp.zeros((), jnp.uint8)).update(
            partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW, overflow
        )
        return idx, err

    def _with_reverse(idx, err, n_atoms):
        # Pair-structure rows computed here once per rebuild instead of once per step
        # (everything else in apax reads only idx[0] / idx[1]):
        #   row 2: reverse-pair slot of every pair (P for padding), used by
        #          neighbor_grad.reverse_pairs (E16).
        # (A row 3 with the CSR starts was 1.7 % slower at 17.9k: the larger idx costs more in
        # the per-step copies around the update cond than the starts kernel it saves; E18.)
        from apax.layers.neighbor_grad import reverse_pairs

        rev, _ = reverse_pairs(idx, n_atoms)
        return jnp.concatenate([idx, rev[None].astype(idx.dtype)], axis=0), err

    def _build(frac, capacity):
        """-> (idx (3, P), err, perm, inv); idx in sorted index space when sort_atoms."""
        N = frac.shape[0]
        if not sort_atoms:
            return (*_with_reverse(*_search(frac, capacity), N), None, None)
        perm, inv = morton_order(frac, species)
        return (*_with_reverse(*_search(frac[perm], capacity), N), perm, inv)

    def _make(idx, position, err, capacity, perm, inv):
        fields = dict(
            idx=idx, reference_position=position, error=err, cell_list_capacity=None,
            max_occupancy=capacity, format=partition.NeighborListFormat.Sparse,
            cell_size=None, cell_list_fn=None, update_fn=update_fn,
        )
        if sort_atoms:
            return SortedNeighborList(**fields, perm=perm, inv=inv)
        return partition.NeighborList(**fields)

    def update_fn(position, nbrs, **kwargs):
        if "box" in kwargs:
            raise ValueError("alchemi neighbour list: constant box only (not NPT)")
        capacity = nbrs.max_occupancy
        d = position - nbrs.reference_position
        d = d - jnp.round(d)
        moved = jnp.sum(_apply3(box32, d.astype(jnp.float32)) ** 2, axis=-1)

        def rebuild(args):
            pos, err = args
            idx, new_err, perm, inv = _build(pos, capacity)
            err = partition.PartitionError(err.code | new_err.code)
            return _make(idx, pos, err, capacity, perm, inv)

        return jax.lax.cond(
            jnp.any(moved > threshold_sq), rebuild, lambda args: nbrs, (position, nbrs.error)
        )

    def allocate_fn(position, extra_capacity=0, **kwargs):
        frac = jnp.asarray(position, jnp.float32)
        N = frac.shape[0]
        mtc, _, nsr = anb.estimate_cell_list_sizes(_cart(frac), cell, cutoff, pbc=pbc)
        out = anb.cell_list(  # eager sizing call
            _cart(frac), cutoff, cell=cell, pbc=pbc, max_total_cells=mtc,
            neighbor_search_radius=nsr, return_neighbor_list=True,
        )
        counts = np.diff(np.asarray(out[1]))  # compact form: (coo, ptr, shifts)
        static.update(
            max_total_cells=int(mtc),
            search_radius=nsr,
            max_neighbors=int(counts.max() * capacity_multiplier) + int(extra_capacity) + 1,
        )
        capacity = int(counts.sum() * capacity_multiplier + N * extra_capacity)
        idx, err, perm, inv = _build(frac, capacity)
        return _make(idx, position, err, capacity, perm, inv)

    return partition.NeighborListFns(allocate_fn, update_fn)  # pytype: disable=wrong-arg-count
