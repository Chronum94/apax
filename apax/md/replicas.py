"""Helpers for running several independent replicas of one system in a single
jitted loop. Every array of the simulation state and of the neighbor list
carries a leading replica axis.
"""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax_md import partition

from apax.utils.space import transform

log = logging.getLogger(__name__)


def replica_slice(tree, k):
    return jax.tree.map(lambda x: x[k], tree)


def add_replica_axis(tree):
    return jax.tree.map(lambda x: jnp.asarray(x)[None], tree)


def select_replicas(mask, tree_true, tree_false):
    """Per replica, take `tree_true` where `mask` is set and `tree_false` elsewhere."""

    def select(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.where(mask.reshape(-1, *[1] * (a.ndim - 1)), a, b)

    return jax.tree.map(select, tree_true, tree_false)


def shard_replicas(tree, n_replicas):
    """Spread the replica axis over all devices, if it divides evenly."""
    n_devices = jax.device_count()
    if n_devices == 1:
        return tree
    if n_replicas % n_devices:
        log.warning(
            "%d replicas do not divide evenly over %d devices; running on one device.",
            n_replicas,
            n_devices,
        )
        return tree
    mesh = Mesh(np.array(jax.devices()), ("replica",))
    return jax.device_put(tree, NamedSharding(mesh, P("replica")))


def _clear_errors(neighbor):
    return dataclasses.replace(
        neighbor, error=partition.PartitionError(jnp.zeros((), jnp.uint8))
    )


def _static_cell_size(neighbor):
    """jax_md stores `cell_size` as a jax array in a static field. Inside a
    trace, comparing it while matching pytree metadata becomes a traced
    operation and fails, so it is kept as a float (it is always a scalar).
    """
    if neighbor.cell_size is None:
        return neighbor
    return dataclasses.replace(neighbor, cell_size=float(neighbor.cell_size))


def allocate_neighbors(
    neighbor_fn, positions, nbr_kwargs, extra_capacity, fractional, max_tries=8
):
    """Allocate one neighbor list per replica, stacked along a leading axis.

    jax_md keeps buffer sizes in static fields, so all replicas share the sizes
    of one template list: the one with the largest cells (smallest NPT box),
    given the largest neighbor and cell capacities of any replica. Every replica
    is then rebuilt from scratch with that template.
    """
    positions = np.asarray(positions)
    n_replicas = len(positions)
    for _ in range(max_tries):
        allocated = [
            _static_cell_size(
                neighbor_fn.allocate(
                    jnp.asarray(positions[k]),
                    extra_capacity=extra_capacity,
                    **replica_slice(nbr_kwargs, k),
                )
            )
            for k in range(n_replicas)
        ]
        if n_replicas == 1:
            return add_replica_axis(allocated[0])

        def largest_cells(neighbor):
            return (
                0.0 if neighbor.cell_size is None else float(np.max(neighbor.cell_size))
            )

        cell_capacities = [
            n.cell_list_capacity for n in allocated if n.cell_list_capacity is not None
        ]
        template = _clear_errors(max(allocated, key=largest_cells))
        max_occupancy = max(n.max_occupancy for n in allocated)
        # padding with the particle count marks unused slots, as jax_md does
        n_pad = max_occupancy - template.idx.shape[-1]
        idx = jnp.pad(
            template.idx,
            [(0, 0)] * (template.idx.ndim - 1) + [(0, n_pad)],
            constant_values=positions.shape[1],
        )
        template = dataclasses.replace(
            template,
            idx=idx,
            max_occupancy=max_occupancy,
            cell_list_capacity=max(cell_capacities) if cell_capacities else None,
        )

        def rebuild(R, kwargs):
            # a reference position far from R forces jax_md to rebuild
            stale = R + 0.5 if fractional else R + 1e6
            neighbor = dataclasses.replace(template, reference_position=stale)
            return neighbor.update(R, **kwargs)

        neighbors = jax.vmap(rebuild)(jnp.asarray(positions), nbr_kwargs)
        if not np.any(neighbors.did_buffer_overflow):
            return neighbors
        extra_capacity = 2 * extra_capacity + 1
        log.info("replica neighbor lists overflow, retrying with more capacity")
    raise RuntimeError("Could not allocate neighbor lists that fit every replica.")


def make_neighbor_update(n_replicas, fractional, dr_threshold=None, r_max=None):
    """Return `update(neighbors, positions, nbr_kwargs, boxes)` for stacked lists.

    Under vmap, jax_md's rebuild `lax.cond` becomes a select that rebuilds every
    replica at every step. Instead, the rebuild runs only when some replica moved
    by more than half the skin. Inside it, jax_md's own check still leaves the
    other replicas unchanged. Without `dr_threshold` the select is used.
    """

    def update_one(neighbor, R, kwargs):
        return neighbor.update(R, **kwargs)

    def update_all(neighbors, positions, nbr_kwargs):
        return jax.vmap(update_one)(neighbors, positions, nbr_kwargs)

    def moved(R, reference, box):
        dR = R - reference
        if fractional:
            dR = transform(box, dR - jnp.round(dR))
        return jnp.any(jnp.sum(dR**2, axis=-1) > (dr_threshold / 2) ** 2)

    def flag_small_cells(neighbor, box):
        # jax_md does this check on every update that gets a box, before deciding
        # whether to rebuild. Skipping the update must not skip the check.
        cutoff = r_max + dr_threshold
        current = partition._cell_size(1.0, neighbor.cell_size)
        new = partition._cell_size(1.0, partition._fractional_cell_size(box, cutoff))
        error = neighbor.error.update(
            partition.PartitionErrorCode.CELL_SIZE_TOO_SMALL, jnp.any(new > current)
        )
        return dataclasses.replace(neighbor, error=error)

    def update(neighbors, positions, nbr_kwargs, boxes):
        if n_replicas == 1:
            # no vmap: keeps jax_md's own lax.cond
            neighbor = replica_slice(neighbors, 0).update(
                positions[0], **replica_slice(nbr_kwargs, 0)
            )
            return add_replica_axis(neighbor)

        if dr_threshold is None:
            return update_all(neighbors, positions, nbr_kwargs)

        def keep(neighbors):
            if "box" in nbr_kwargs and neighbors.cell_size is not None:
                if r_max is None:
                    return update_all(neighbors, positions, nbr_kwargs)
                return jax.vmap(flag_small_cells)(neighbors, nbr_kwargs["box"])
            return neighbors

        any_moved = jnp.any(
            jax.vmap(moved)(positions, neighbors.reference_position, boxes)
        )
        return jax.lax.cond(
            any_moved,
            lambda n: update_all(n, positions, nbr_kwargs),
            keep,
            neighbors,
        )

    return update
