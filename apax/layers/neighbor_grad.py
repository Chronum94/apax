"""Neighbour gather with a scatter-free backward pass, for jax-md sparse neighbour lists.

`compute_distances` gathers neighbour positions with R[idx[0]]. Its backward pass is an
unsorted scatter-add of every pair's cotangent into the neighbour atom: on GPUs a heavily
contended atomic scatter (each atom is the neighbour of ~100 others) and the most expensive
kernel of an MD force call (0.5 ms of a ~1.5 ms step at 3.9k atoms, 1.9 ms at 17.9k on an
L40S).

jax-md's `Sparse` lists are full (every pair (b, a) has its reverse (a, b)), sorted by centre
idx[1], and, under the minimum-image check, free of duplicate pairs. The cotangent of R[b]
from the pairs where b is the neighbour is then

    sum_{p: nbr(p) = b} ct[p] = sum_{q in segment(b)} ct[rev(q)],

a gather from each pair's reverse slot followed by a sorted segment sum: no atomics.
rev is found by a Warp kernel (each pair scans its neighbour's segment). Results equal the
scatter up to fp32 summation order.

Only valid for symmetric, centre-sorted lists without duplicate pairs (jax-md Sparse with
the minimum-image check); not for OrderedSparse (half) lists or training batches.
"""

import functools

import jax
import jax.numpy as jnp


@functools.lru_cache(maxsize=None)
def _reverse_kernel():
    import warp as wp

    @wp.kernel(module="unique")
    def reverse_pairs(
        starts: wp.array(dtype=wp.int32),
        pairs: wp.array2d(dtype=wp.int32),  # row 0 neighbour, row 1 centre
        n_atoms: wp.int32,
        rev: wp.array(dtype=wp.int32),
    ):
        # rev is an involution: search only from the pair with centre < neighbour and
        # write both directions (each slot has exactly one writer). rev is pre-filled with -1
        # (padding / pairs without a reverse, e.g. jax-md's stray last slot).
        p = wp.tid()
        b = pairs[1, p]
        a = pairs[0, p]
        if b < a and a < n_atoms:
            for q in range(starts[a], starts[a + 1]):
                if pairs[0, q] == b:
                    rev[p] = q
                    rev[q] = p
                    break

    return wp.jax_kernel(reverse_pairs, num_outputs=1, in_out_argnames=["rev"])


@functools.lru_cache(maxsize=None)
def _starts_kernel():
    import warp as wp

    @wp.func
    def cval(pairs: wp.array2d(dtype=wp.int32), i: wp.int32, n_pairs: wp.int32, n_atoms: wp.int32):
        c = wp.min(pairs[1, i], n_atoms)
        if i == n_pairs - 1 and n_pairs > 1:
            if pairs[1, i] < pairs[1, i - 1]:
                c = n_atoms
        return c

    @wp.kernel(module="unique")
    def segment_starts(
        pairs: wp.array2d(dtype=wp.int32),  # row 1 = centre
        n_pairs: wp.int32,
        n_atoms: wp.int32,
        starts: wp.array(dtype=wp.int32),
    ):
        # thread p in [0, P]: every atom a with centre[p-1] < a <= centre[p] starts at p
        # (centre sorted; padding = n_atoms; p = P closes the list). Each starts[a] is
        # written exactly once.
        p = wp.tid()
        prev = wp.int32(-1)
        if p > 0:
            prev = cval(pairs, p - 1, n_pairs, n_atoms)
        cur = n_atoms
        if p < n_pairs:
            cur = cval(pairs, p, n_pairs, n_atoms)
        for a in range(prev + 1, cur + 1):
            starts[a] = p

    return wp.jax_kernel(segment_starts, num_outputs=1)


@functools.lru_cache(maxsize=None)
def _reduce_kernel(n_split):
    import warp as wp

    S = n_split

    @wp.kernel(module="unique")
    def pair_grad_reduce(
        starts: wp.array(dtype=wp.int32),
        rev: wp.array2d(dtype=wp.int32),  # reverse slots in row rev_row
        rev_row: wp.int32,
        ct_c: wp.array(dtype=wp.vec3),
        ct_n: wp.array(dtype=wp.vec3),
        n_pairs: wp.int32,
        dR: wp.array(dtype=wp.vec3),  # in/out, zero-initialised
    ):
        # dR[b] = sum over b's segment of ct_c[q] + ct_n[rev[q]]; S threads per atom
        b, s = wp.tid()
        acc = wp.vec3()
        for q in range(starts[b] + s, starts[b + 1], S):
            acc += ct_c[q]
            r = rev[rev_row, q]
            if r < n_pairs:
                acc += ct_n[r]
        wp.atomic_add(dR, b, acc)

    return wp.jax_kernel(pair_grad_reduce, num_outputs=1, in_out_argnames=["dR"])


def centre_csr(idx, n_atoms):
    """(pairs (2, P) int32, starts (N + 1,)) of a centre-sorted jax-md sparse list. jax-md writes every
    pruned-away entry to the last slot, which can then hold a stray out-of-order pair
    (>= the list cutoff apart): `starts` treats it as padding (no segment contains it).
    The centre row is left unmodified (an .at[-1].set here made XLA copy the whole row,
    ~10 MB per call at 17.9k atoms); kernels only reach pairs through `starts`, or give
    the stray (beyond-cutoff, zero-radial) pair zero contribution."""
    # The Warp kernels take the whole (rows, P) array and index the row themselves: a row
    # slice as a custom-call operand is materialised by XLA (a ~9 MB device copy per call at
    # 17.9k atoms). (In session 2 this measured 5.6 % slower through side effects on XLA's
    # contraction kernels; with contract_kernel: warp it is a gain, E19.)
    pairs = idx.astype(jnp.int32)
    # segment starts written where the sorted centre index changes (Warp, P + 1 threads).
    # Not jnp.searchsorted (an XLA while loop: host round trips) and not a segment_sum of
    # ones + cumsum (an int scatter over all pairs: ~18 us at 3.9k atoms on an L40S).
    P = pairs.shape[1]
    (starts,) = _starts_kernel()(
        pairs, int(P), int(n_atoms), launch_dims=(P + 1,), output_dims=(n_atoms + 1,)
    )
    return pairs, starts


def reverse_pairs(idx, n_atoms, with_starts=False):
    """rev[p] = slot of the reverse pair of p, or P (out of range) for padding.
    A third row of idx, if present, is taken as rev: the ALCHEMI neighbour-list adapter
    computes it once per rebuild instead of once per step (E16)."""
    pairs, starts = centre_csr(idx, n_atoms)
    P = pairs.shape[1]
    if pairs.shape[0] >= 3:
        rev = pairs[2]
    else:
        (rev,) = _reverse_kernel()(
            starts, pairs, int(n_atoms), jnp.full((P,), -1, jnp.int32), launch_dims=(P,),
        )
        rev = jnp.where(rev < 0, jnp.int32(P), rev)
    if with_starts:
        return rev, pairs, starts
    return rev, pairs


@jax.custom_vjp
def gather_pair(R, idx):
    """(R[idx[1]], R[idx[0]]): centre and neighbour positions of every pair, with one
    scatter-free backward pass for both: dR[b] = sum_{q in segment(b)} ct_c[q] + ct_n[rev(q)]
    (a single sorted segment sum instead of two). The centre gather clamps like
    jnp.take(..., mode="clip"); padding / stray-slot cotangents are dropped (they are zero:
    those pairs are masked or beyond the cutoff)."""
    return (
        jnp.take(R, idx[1], axis=0, mode="clip", indices_are_sorted=True),
        R[idx[0]],
    )


def _pair_fwd(R, idx):
    return gather_pair(R, idx), (idx, jnp.zeros((R.shape[0], 0), R.dtype))


def _pair_bwd(res, cts):
    idx, like = res
    ct_c, ct_n = cts
    n_atoms, dtype = like.shape[0], like.dtype
    rev, pairs, starts = reverse_pairs(idx, n_atoms, with_starts=True)
    P = pairs.shape[1]
    if ct_c.dtype != jnp.float32 or ct_n.dtype != jnp.float32:
        # fp64 paths (e.g. ASE / trajectory evaluation): keep full precision. The stray
        # last slot (if any) carries a zero cotangent (beyond the cutoff).
        total = ct_c + jnp.take(ct_n, rev, axis=0, mode="fill", fill_value=0)
        grad_R = jax.ops.segment_sum(total, pairs[1], n_atoms)
        return grad_R.astype(dtype), None
    # segmented reduction over each atom's contiguous pair run (Warp, S threads per atom)
    # instead of take + add + XLA sorted segment_sum (one ~50 us scatter kernel at 3.9k atoms)
    S = 4
    zeros = jnp.zeros((n_atoms, 3), jnp.float32)
    # reverse slots: cached row 2 of the pair array (ALCHEMI lists) or the computed vector
    rev_src, rev_row = (pairs, 2) if pairs.shape[0] >= 3 else (rev[None], 0)
    (grad_R,) = _reduce_kernel(S)(
        starts, rev_src, rev_row, ct_c.astype(jnp.float32), ct_n.astype(jnp.float32), int(P),
        zeros, launch_dims=(n_atoms, S),
    )
    return grad_R.astype(dtype), None


gather_pair.defvjp(_pair_fwd, _pair_bwd)



# --- fused periodic pair vectors (E21) -------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _pair_vector_kernels(n_split):
    import warp as wp

    S = n_split

    @wp.kernel(module="unique")
    def pair_vectors(
        R: wp.array(dtype=wp.vec3),  # fractional positions
        pairs: wp.array2d(dtype=wp.int32),  # row 0 neighbour, row 1 centre
        box: wp.array(dtype=wp.mat33),  # (1,): lattice vectors as columns
        n_atoms: wp.int32,
        dr: wp.array(dtype=wp.vec3),
    ):
        # dr = box @ (s - floor(s + 1/2)), s = R[centre] - R[neighbour]: the minimum-image
        # displacement of periodic_general_elementwise (mod(s + 1/2, 1) - 1/2). Padding (index
        # >= N) gives 0, as the clamped gathers of the XLA path (both ends at atom N - 1).
        p = wp.tid()
        c = wp.min(pairs[1, p], n_atoms - 1)
        n = wp.min(pairs[0, p], n_atoms - 1)
        s = R[c] - R[n]
        h = wp.float32(0.5)
        s = wp.vec3(s[0] - wp.floor(s[0] + h), s[1] - wp.floor(s[1] + h), s[2] - wp.floor(s[2] + h))
        dr[p] = box[0] @ s

    @wp.kernel(module="unique")
    def pair_vector_grad(
        starts: wp.array(dtype=wp.int32),
        rev: wp.array2d(dtype=wp.int32),  # reverse slots in row rev_row
        rev_row: wp.int32,
        ct: wp.array(dtype=wp.vec3),
        n_pairs: wp.int32,
        dR: wp.array(dtype=wp.vec3),  # in/out, zero-initialised
    ):
        # d dr / dR[centre] = +1, d dr / dR[neighbour] = -1 (fractional tangents pass through
        # unscaled: apax/jax-md convention, gradients come back in real space), so
        # dR[b] = sum over b's segment of ct[q] - ct[rev[q]]; S threads per atom
        b, s = wp.tid()
        acc = wp.vec3()
        for q in range(starts[b] + s, starts[b + 1], S):
            acc += ct[q]
            r = rev[rev_row, q]
            if r < n_pairs:
                acc -= ct[r]
        wp.atomic_add(dR, b, acc)

    fwd = wp.jax_kernel(pair_vectors, num_outputs=1)
    bwd = wp.jax_kernel(pair_vector_grad, num_outputs=1, in_out_argnames=["dR"])
    return fwd, bwd


@jax.custom_vjp
def pair_vectors(R, idx, box):
    """Minimum-image pair vectors box @ wrap(R[idx[1]] - R[idx[0]]) of fractional positions
    R (N, 3) fp32, a centre-sorted symmetric list idx and a constant box (3, 3): the same
    values as gather_pair + periodic_general_elementwise in compute_distances, in one Warp
    kernel each way (no pair-sized XLA gathers, wraps, box products or their transposes).
    The box gets no cotangent: callers use it only for constant boxes."""
    fwd, _ = _pair_vector_kernels(4)
    n_atoms, P = R.shape[0], idx.shape[1]
    (dr,) = fwd(
        R.astype(jnp.float32), idx.astype(jnp.int32), jnp.asarray(box, jnp.float32)[None],
        int(n_atoms), launch_dims=(P,), output_dims=(P, 3),
    )
    return dr


def _pv_fwd(R, idx, box):
    return pair_vectors(R, idx, box), (idx, jnp.zeros((R.shape[0], 0), R.dtype), box)


def _pv_bwd(res, ct):
    idx, like, box = res
    n_atoms = like.shape[0]
    rev, pairs, starts = reverse_pairs(idx, n_atoms, with_starts=True)
    P = pairs.shape[1]
    rev_src, rev_row = (pairs, 2) if pairs.shape[0] >= 3 else (rev[None], 0)
    S = 4
    _, bwd = _pair_vector_kernels(S)
    (dR,) = bwd(
        starts, rev_src, rev_row, ct.astype(jnp.float32), int(P),
        jnp.zeros((n_atoms, 3), jnp.float32), launch_dims=(n_atoms, S),
    )
    return dR.astype(like.dtype), None, jnp.zeros_like(box)


pair_vectors.defvjp(_pv_fwd, _pv_bwd)
