import einops
import jax
import numpy as np

# Independent components of the symmetric rank-2 / rank-3 moment tensors,
# with their multiplicity weights (off-diagonal entries appear more than
# once in the full tensor). Any full contraction over the symmetric spatial
# axes reduces to a weighted dot over just these components.
_P2 = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
_P3 = [
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2),
    (0, 2, 2), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2),
]
_IDX2 = np.array(_P2).T  # (2, 6)
_IDX3 = np.array(_P3).T  # (3, 10)


def _unpack_index_map(rank, packed):
    """(3,)*rank array mapping each spatial index tuple to its packed slot."""
    shape = (3,) * rank
    m = np.zeros(shape, dtype=np.int32)
    for idx in np.ndindex(shape):
        m[idx] = packed.index(tuple(sorted(idx)))
    return m


_UNPACK2 = _unpack_index_map(2, _P2)  # (3, 3)
_UNPACK3 = _unpack_index_map(3, _P3)  # (3, 3, 3)


def packed_geometric_moments(radial_function, dn, idx_i, n_atoms=None):
    """Same signature/return contract as geometric_moments. Accumulates the
    rank-2/rank-3 moments in their 6/10 independent components (instead of
    the full, redundant 9/27) via segment_sum, then unpacks back to the full
    symmetric tensors once per atom (not once per neighbor) -- so this is a
    smaller-memory-traffic drop-in, and downstream contraction code is
    unaffected."""
    # dn shape: neighbors x 3, radial_function shape: n_neighbors x n_radial
    x2 = dn[:, _IDX2[0]] * dn[:, _IDX2[1]]                    # neighbors x 6
    x3 = dn[:, _IDX3[0]] * dn[:, _IDX3[1]] * dn[:, _IDX3[2]]  # neighbors x 10

    seg = lambda arr: jax.ops.segment_sum(arr, idx_i, n_atoms)
    f = radial_function

    zero_moment = seg(f)
    first_moment = seg(f[..., None] * dn[:, None, :])
    u_packed = seg(f[..., None] * x2[:, None, :])
    v_packed = seg(f[..., None] * x3[:, None, :])

    second_moment = u_packed[..., _UNPACK2]
    third_moment = v_packed[..., _UNPACK3]

    return [zero_moment, first_moment, second_moment, third_moment]


def geometric_moments(radial_function, dn, idx_i, n_atoms=None):
    # dn shape: neighbors x 3
    # radial_function shape: n_neighbors x n_radial

    # s = spatial dim = 3
    xyz = einops.repeat(dn, "nbrs s -> nbrs 1 s")
    xyz2 = einops.repeat(dn, "nbrs s -> nbrs 1 1 s")
    xyz3 = einops.repeat(dn, "nbrs s -> nbrs 1 1 1 s")

    # shape: n_neighbors x n_radial x (3)^(moment_number)
    # s_i = spatial = 3
    zero_moment = radial_function
    first_moment = einops.repeat(zero_moment, "n r -> n r 1") * xyz
    second_moment = einops.repeat(first_moment, "n r s1 -> n r s1 1") * xyz2
    third_moment = einops.repeat(second_moment, "n r s1 s2 -> n r s1 s2 1") * xyz3

    # shape: n_atoms x n_radial x (3)^(moment_number)
    zero_moment = jax.ops.segment_sum(zero_moment, idx_i, n_atoms)
    first_moment = jax.ops.segment_sum(first_moment, idx_i, n_atoms)
    second_moment = jax.ops.segment_sum(second_moment, idx_i, n_atoms)
    third_moment = jax.ops.segment_sum(third_moment, idx_i, n_atoms)

    moments = [zero_moment, first_moment, second_moment, third_moment]

    return moments
