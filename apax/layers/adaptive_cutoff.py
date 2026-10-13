"""Adaptive per-atom cutoffs, ported from pet-jax (``petjax/select.py``).

Each atom gets a cutoff ``r_i`` such that its smoothly counted number of
neighbors matches ``num_neighbors``. A pair is kept iff ``r_ij <= (r_i + r_j) / 2``
and is weighted by ``cutoff_bump(r_ij; pair_cutoff, cutoff_width)``, which vanishes
at the selection boundary, so selection is continuous. The raw neighbor list is
built at the maximum cutoff (``basis.r_max``); selected edges are packed into a
flat ``[n_atoms * k_sel]`` edge list, bounding the per-atom environment size.

The per-atom cutoffs depend on the positions of atoms that end up *outside* the
final cutoff, so gradients must flow through them ("shadow forces"). The solver
iterates on detached distances and re-attaches gradients via a single
implicit-function-theorem step.

Padding convention is apax's: a pair with ``idx_i == idx_j`` and zero
displacement is padding; with a nonzero displacement it is a periodic self-image.
"""

import functools
import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from apax.layers.distances import canonicalize_neighbors, make_distance_fn

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveCutoff:
    """Selection + packing, used inside ``EnergyModel``/``FeatureModel``.

    ``k_sel`` is a shape parameter (static under jit), not a hyperparameter: it is
    sized from the data at hand and never saved with the model.
    """

    num_neighbors: int
    cutoff: float
    cutoff_width_adaptive: float = 0.5
    cutoff_width: float = 0.5
    k_sel: int = 0
    # Cut gradients through the per-atom cutoffs (drops the shadow forces, which
    # makes forces non-conservative). For testing/diagnostics only.
    no_shadow: bool = False

    def __call__(self, dr_vec, idx, n_atoms):
        """Returns ``(dr_vec, idx, weights, overflow)`` on the packed edge list."""
        _, pair_cutoffs, selected, _ = select_edges(
            dr_vec,
            idx,
            n_atoms,
            self.num_neighbors,
            self.cutoff,
            self.cutoff_width_adaptive,
            no_shadow=self.no_shadow,
        )
        r_ij = safe_norm(dr_vec)
        weights = cutoff_bump(r_ij, pair_cutoffs, self.cutoff_width)
        return pack_edges(dr_vec, idx, selected, weights, n_atoms, self.k_sel)


def build_adaptive_cutoff(config: dict, k_sel: int):
    """``AdaptiveCutoff`` from a model config dict, or None if not enabled."""
    adaptive = config.get("adaptive_cutoff")
    if adaptive is None:
        return None
    if k_sel is None:
        raise ValueError(
            "model uses adaptive_cutoff: k_sel must be sized from the data "
            "(see apax.layers.adaptive_cutoff.size_k_sel) and passed to the builder"
        )
    return AdaptiveCutoff(
        num_neighbors=adaptive["num_neighbors"],
        cutoff=config["basis"]["r_max"],
        cutoff_width_adaptive=adaptive["cutoff_width_adaptive"],
        cutoff_width=adaptive["cutoff_width"],
        k_sel=int(k_sel),
    )


def size_k_sel(max_count, extra_neighbors):
    return max(int(max_count), 1) + int(extra_neighbors)


def k_sel_for_inputs(config: dict, *inputs) -> int | None:
    """``k_sel`` covering all structures in the given apax input dicts, or None if
    the model does not use adaptive cutoffs."""
    adaptive = config.get("adaptive_cutoff")
    if adaptive is None:
        return None
    max_count = max(max_selected_count(config, inp) for inp in inputs)
    k_sel = size_k_sel(max_count, adaptive["extra_neighbors"])
    log.info(f"Adaptive cutoff: max selected neighbors {max_count}, k_sel = {k_sel}")
    return k_sel


def make_count_fn(config: dict, init_box, inference_disp_fn=None):
    """Parameter-free ``(R, Z, neighbor, box, offsets) -> counts`` for sizing
    ``k_sel`` and for overflow checks: selected pairs per atom, under the same
    distance conventions as the model."""
    adaptive = config["adaptive_cutoff"]
    num_neighbors = adaptive["num_neighbors"]
    cutoff = config["basis"]["r_max"]
    width = adaptive["cutoff_width_adaptive"]
    compute_distances = make_distance_fn(np.asarray(init_box), inference_disp_fn)

    def count_fn(R, Z, neighbor, box, offsets):
        dr_vec, idx = compute_distances(R, neighbor, box, offsets)
        *_, counts = select_edges(dr_vec, idx, Z.shape[0], num_neighbors, cutoff, width)
        return counts

    return count_fn


def max_selected_count(config: dict, inputs: dict, nls=None) -> int:
    """Largest number of selected neighbors over a list of structures, on CPU.

    ``inputs`` is apax's unpadded input dict (Cartesian ``positions``, ``numbers``,
    ``box``), ``nls`` the matching ``(idx, offsets)``, which
    are computed at ``basis.r_max`` if not given. Structures are padded to coarse
    size buckets, so that the counting kernel compiles only a few times.
    """
    from apax.data.preprocessing import compute_nl

    adaptive = config["adaptive_cutoff"]
    cpu = jax.devices("cpu")[0]
    max_count = 0
    for i in range(len(inputs["positions"])):
        R = np.asarray(inputs["positions"][i])
        Z = np.asarray(inputs["numbers"][i])
        box = np.asarray(inputs["box"][i])
        if nls is None:
            idx, offsets = compute_nl(R, box, config["basis"]["r_max"])
        else:
            idx, offsets = nls[i]
        idx = np.asarray(idx, dtype=np.int32)
        offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)

        n_atoms, n_pairs = len(Z), idx.shape[1]
        n_atoms_pad = _bucket(n_atoms + 1, 16) - n_atoms
        n_pairs_pad = _bucket(n_pairs + 1, 256) - n_pairs
        R = np.pad(R, ((0, n_atoms_pad), (0, 0)))
        Z = np.pad(Z, (0, n_atoms_pad))
        idx = np.pad(idx, ((0, 0), (0, n_pairs_pad)))
        offsets = np.pad(offsets, ((0, n_pairs_pad), (0, 0)))

        count_fn = _count_kernel(
            adaptive["num_neighbors"],
            config["basis"]["r_max"],
            adaptive["cutoff_width_adaptive"],
        )
        with jax.default_device(cpu):
            counts = count_fn(R, Z, idx, box, offsets)
        max_count = max(max_count, int(np.max(counts, initial=0)))
    return max_count


@functools.lru_cache(maxsize=16)
def _count_kernel(num_neighbors, r_max, cutoff_width_adaptive):
    """Jitted counting kernel for host-side sizing (Cartesian inputs), shared
    across calls so it compiles once per size bucket rather than once per call."""
    config = {
        "basis": {"r_max": r_max},
        "adaptive_cutoff": {
            "num_neighbors": num_neighbors,
            "cutoff_width_adaptive": cutoff_width_adaptive,
        },
    }
    return jax.jit(make_count_fn(config, np.zeros(3)))


def _bucket(n, minimum):
    """Round up to ``minimum`` or the next power of two over 1.25 steps."""
    n = max(int(n), minimum)
    step = 2 ** max(int(np.log2(n)) - 2, 0)
    return int(np.ceil(n / step) * step)


# -- selection --


def select_edges(
    dr_vec, idx, n_atoms, num_neighbors, cutoff, cutoff_width_adaptive, no_shadow=False
):
    """Returns ``(atomic_cutoffs, pair_cutoffs, selected, counts)``."""
    idx = jnp.asarray(canonicalize_neighbors(idx))
    centers, others = idx[0], idx[1]
    # Padding is (i, i) with zero displacement; (i, i) with a nonzero one is an
    # atom's own periodic image (as in masking.mask_by_neighbor).
    pair_mask = (centers != others) | (jnp.sum(dr_vec**2, axis=-1) > 1e-12)
    r_ij = safe_norm(dr_vec)

    atomic_cutoffs = get_adaptive_cutoffs_solver(
        centers, r_ij, pair_mask, num_neighbors, n_atoms, cutoff, cutoff_width_adaptive
    )
    if no_shadow:
        atomic_cutoffs = jax.lax.stop_gradient(atomic_cutoffs)
    # jax-md pads with index n_atoms; the clamped gather is harmless (masked).
    pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[others]) / 2
    selected = (r_ij <= pair_cutoffs) & pair_mask
    counts = jax.ops.segment_sum(selected.astype(jnp.int32), centers, n_atoms)
    return atomic_cutoffs, pair_cutoffs, selected, counts


# -- packing: selected edges -> flat [n_atoms * k_sel] --


def pack_edges(dr_vec, idx, selected, weights, n_atoms, k_sel):
    """Compact selected edges into ``n_atoms * k_sel`` slots, at most ``k_sel``
    per center. Unused slots are padding: ``idx = (0, 0)``, ``dr = 0``, ``w = 0``.

    The raw list need not be sorted: a stable argsort groups selected edges by
    center (unselected ones go last), then pet-jax's cumsum trick assigns the
    column within each row. ``overflow`` is True iff a selected edge was dropped.
    """
    idx = jnp.asarray(canonicalize_neighbors(idx))
    centers = idx[0]
    n_slots = n_atoms * k_sel

    key = jnp.where(selected, centers, n_atoms)
    order = jnp.argsort(key, stable=True)
    sel_sorted = selected[order]
    centers_sorted = key[order]  # n_atoms for unselected

    sel_int = sel_sorted.astype(jnp.int32)
    count = jax.ops.segment_sum(sel_int, centers_sorted, n_atoms + 1)
    prefix = jnp.concatenate([jnp.zeros(1, count.dtype), jnp.cumsum(count)[:-1]])
    column = jnp.cumsum(sel_int) - 1 - prefix[centers_sorted]

    fits = sel_sorted & (column < k_sel)
    overflow = jnp.any(sel_sorted & (column >= k_sel))

    # Non-fitting edges scatter to an out-of-bounds slot and are dropped.
    slot = jnp.where(fits, centers_sorted * k_sel + column, n_slots)
    slot_to_pair = (
        jnp.zeros(n_slots, dtype=jnp.int32)
        .at[slot]
        .set(order.astype(jnp.int32), mode="drop")
    )
    slot_mask = jnp.zeros(n_slots, dtype=bool).at[slot].set(True, mode="drop")

    idx_p = jnp.where(slot_mask[None, :], idx[:, slot_to_pair], 0).astype(idx.dtype)
    dr_p = jnp.where(slot_mask[:, None], dr_vec[slot_to_pair], 0.0)
    w_p = jnp.where(slot_mask, weights[slot_to_pair], 0.0)
    return dr_p, idx_p, w_p, overflow


# -- adaptive per-atom cutoffs (solver), verbatim port from pet-jax --


def get_adaptive_cutoffs_solver(
    centers, r_ij, pair_mask, num_neighbors, num_atoms, cutoff, cutoff_width
):
    """Per-atom adaptive cutoffs via a Newton-bisection root find on the
    smoothed neighbor count (metatrain's "solver" method). Solves
    ``n_total(r) = num_neighbors`` per atom, where
    ``n_total(r) = sum_j bump(r_j, r, w) + num_neighbors * (r / cutoff)**3``;
    the cubic baseline makes ``n_total`` monotonic so the root is unique and
    bracketed by ``[0, cutoff]``.

    The iteration runs on gradient-detached distances; the trailing
    implicit-function-theorem step re-attaches gradients through the residual,
    so the backward never differentiates through the solver loop. Note that this
    makes first derivatives exact, but not second derivatives.
    """
    inv_cutoff = 1.0 / cutoff
    r_ij_d = jax.lax.stop_gradient(r_ij)

    # Under shard_map, fori_loop needs the carry typed "varying" like r_ij;
    # pcast re-tags the constant bracket. No-op outside shard_map.
    varying = tuple(jax.typeof(r_ij_d).manual_axis_type.varying)
    r_lo = jax.lax.pcast(jnp.zeros(num_atoms, dtype=r_ij.dtype), varying, to="varying")
    r_hi = jax.lax.pcast(
        jnp.full(num_atoms, cutoff, dtype=r_ij.dtype), varying, to="varying"
    )

    # 10 iterations converge to fp32 precision (upstream's choice). Newton steps
    # that would leave the bracket fall back to the bracket midpoint.
    def newton_bisection_step(_, carry):
        r_lo, r_hi, r = carry
        n, dn = _n_total_and_dn_dr(
            r,
            r_ij_d,
            centers,
            pair_mask,
            num_atoms,
            cutoff_width,
            inv_cutoff,
            num_neighbors,
        )
        f = n - num_neighbors
        below = f <= 0
        r_lo = jnp.where(below, r, r_lo)
        r_hi = jnp.where(below, r_hi, r)
        r_newton = r - f / jnp.clip(dn, 1e-6, None)
        inside = (r_newton >= r_lo) & (r_newton <= r_hi)
        return r_lo, r_hi, jnp.where(inside, r_newton, 0.5 * (r_lo + r_hi))

    _, _, r = jax.lax.fori_loop(0, 10, newton_bisection_step, (r_lo, r_hi, 0.5 * r_hi))
    _, dn_root = _n_total_and_dn_dr(
        r, r_ij_d, centers, pair_mask, num_atoms, cutoff_width, inv_cutoff, num_neighbors
    )

    # IFT step: r and dn_root are constants; gradients attach only through the
    # residual (live r_ij). Clamps mirror upstream: the derivative floor bounds
    # the correction in pathological geometries, cutoff/16 is the lower bound.
    n_residual = (
        _n_total(
            r,
            r_ij,
            centers,
            pair_mask,
            num_atoms,
            cutoff_width,
            inv_cutoff,
            num_neighbors,
        )
        - num_neighbors
    )
    return jnp.clip(r - n_residual / jnp.clip(dn_root, 1e-6, None), cutoff / 16, cutoff)


def _n_total(
    r_per_atom,
    r_ij,
    centers,
    pair_mask,
    num_atoms,
    cutoff_width,
    inv_cutoff,
    num_neighbors,
):
    """Smoothed neighbor count plus cubic baseline at per-atom cutoff ``r_per_atom``."""
    per_edge = cutoff_bump(r_ij, r_per_atom[centers], cutoff_width) * pair_mask
    n = jax.ops.segment_sum(per_edge, centers, num_atoms)
    x = r_per_atom * inv_cutoff
    return n + num_neighbors * x**3


def _n_total_and_dn_dr(
    r_per_atom,
    r_ij,
    centers,
    pair_mask,
    num_atoms,
    cutoff_width,
    inv_cutoff,
    num_neighbors,
):
    """``n_total`` and its analytic d/dr in one pass, for the Newton steps.

    Closed form of the bump in its active region ``s = (d - r + w)/w`` in (0, 1):
    ``f = 0.5 * (1 + tanh(cot(pi s)))``, ``df/dr = (pi / 2w) sech^2(cot(pi s)) /
    sin^2(pi s)``; outside, f saturates to 1 (below) / 0 (above) with df/dr = 0.
    """
    scaled = (r_ij - (r_per_atom[centers] - cutoff_width)) / cutoff_width
    active = (scaled > 0.0) & (scaled < 1.0) & pair_mask
    smaller = (scaled <= 0.0) & pair_mask

    safe = jnp.clip(scaled, 1e-6, 1 - 1e-6)
    s = jnp.pi * safe
    sin_s = jnp.sin(s)
    tanh_cot = jnp.tanh(jnp.cos(s) / sin_s)

    f = jnp.where(active, 0.5 * (1.0 + tanh_cot), smaller.astype(scaled.dtype))
    df_dr = jnp.where(
        active,
        (0.5 * jnp.pi / cutoff_width) * (1.0 - tanh_cot**2) / (sin_s * sin_s),
        0.0,
    )

    n = jax.ops.segment_sum(f, centers, num_atoms)
    dn = jax.ops.segment_sum(df_dr, centers, num_atoms)

    x = r_per_atom * inv_cutoff
    n = n + num_neighbors * x**3
    dn = dn + 3.0 * num_neighbors * x**2 * inv_cutoff
    return n, dn


# -- utilities --


def cutoff_bump(r, cutoff, width=0.5):
    """Smooth bump: 1 below ``cutoff - width``, 0 at and above ``cutoff``."""
    x = (r - (cutoff - width)) / width
    x_safe = jnp.clip(x, 1e-6, 1 - 1e-6)
    bump = 0.5 * (1 + jnp.tanh(1 / jnp.tan(jnp.pi * x_safe)))
    return jnp.where(x <= 0, 1.0, jnp.where(x >= 1, 0.0, bump))


def safe_norm(x, axis=-1, eps=1e-15):
    # +eps keeps the gradient of sqrt finite for zero-length (padding) edges
    return jnp.sqrt(jnp.sum(jax.lax.square(x), axis=axis) + eps)
