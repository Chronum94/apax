"""Fused Gaussian-moment contractions (NVIDIA Warp) for inference (E15).

Replaces `GaussianMomentDescriptor._contract` on the `moment_kernel="warp"` path: from the
packed per-atom moments of `warp_moments.py` (atom-major rows of K + 19 C floats:
m0 (K) | m1 (C, 3) | p2 (C, 6 unique of the symmetric 3x3) | p3 (C, 10 unique of 3x3x3))
to the contraction features contr_1 ... contr_7, in exactly the order `_contract` produces
them (contr_0 = m0 stays a slice in XLA).

Why: in XLA the contractions are ~30 einsums, tril gathers, unpacking stacks and their
transposes: 1.3 ms of a 2.4 ms MD step at 17.9k atoms (L40S), memory-bound and sensitive to
XLA's kernel order (L2 reuse). Per atom they are tiny multilinear forms of ~107 numbers, so
one kernel each way, with everything in registers, is enough.

Each feature is one of seven multilinear families (b = m1[c] vectors, C = m2[c] symmetric
3x3, D = m3[c] symmetric 3x3x3 as three 3x3 slices D[i]_jk; c0, c1, c2 the channel indices):

  1  b0 . b1                         (ari, asi -> ars)
  2  <C0, C1>                        (arij, asij -> ars)
  3  sum_i <D0[i], D1[i]>            (arijk, asijk -> ars)
  4  <C0, C1 C2^T>                   (arij, asik, atjk -> arst)
  5  b0^T C2 b1                      (ari, asj, atij -> arst)
  6  sum_i <D0[i] C2, D1[i]>         (arijk, asijl, atkl -> arst)
  7  sum_i row_i(C1) . (D0[i] b2)    (arijk, asij, atk -> arst)

A static table (one row per feature: family, c0, c1, c2) is built in Python from the same
`tril_2d_indices` / `tril_3d_indices` and `n_contr` / `drop_contractions` as `_contract`.

  * forward: one thread per (atom, feature), no atomics;
  * backward (hand-derived VJP, the forms are multilinear): the static work list of all
    (feature, slot) occurrences is grouped by (channel, family, slot position) into chunks of
    <= 32 features (`backward_chunks`); one thread per (chunk, atom), atoms fastest, so a warp
    never diverges. A thread accumulates the full-tensor gradient of one tensor of one channel
    in registers, folds it onto the packed unique components and adds them with 3 / 6 / 10
    atomics. The m0 slots stay zero (contr_0 is the identity, differentiated by XLA).
    (A first version with one thread per (atom, channel) walking all features took 6.9 ms at
    17.9k atoms: divergent lanes, low parallelism.)

fp32 throughout, as the XLA path; results match it to fp32 round-off (summation order).
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from apax.layers.descriptor.triangular_indices import tril_2d_indices, tril_3d_indices


def feature_table(C, n_contr=8, drop_contractions=()):
    """(n_features, 4) int32 rows (family, c0, c1, c2) for contr_1..7 in `_contract` order."""
    with (
        jax.ensure_compile_time_eval()
    ):  # the helpers build jnp arrays; keep them concrete
        t2 = np.asarray(tril_2d_indices(C))
        t3 = np.asarray(tril_3d_indices(C))
    blocks = {
        1: [(1, i, j, 0) for i, j in t2],
        2: [(2, i, j, 0) for i, j in t2],
        3: [(3, i, j, 0) for i, j in t2],
        4: [(4, i, j, k) for i, j, k in t3],
        # contr_5/6[:, tril_i, tril_j] (n_atoms, n_pairs, C) reshaped: pair-major, t minor
        5: [(5, i, j, t) for i, j in t2 for t in range(C)],
        6: [(6, i, j, t) for i, j in t2 for t in range(C)],
        7: [(7, r, s, t) for r in range(C) for s in range(C) for t in range(C)],
    }
    rows = []
    for k in range(1, min(int(n_contr), 8)):
        if k not in drop_contractions:
            rows += blocks[k]
    return np.asarray(rows, np.int32).reshape(-1, 4)


@functools.lru_cache(maxsize=None)
def _kernels(K, C):
    import warp as wp

    B1 = K  # m1 offset
    B2 = K + 3 * C  # p2 offset
    B3 = K + 9 * C  # p3 offset

    @wp.func
    def vec_b(m: wp.array2d(dtype=wp.float32), a: int, c: int):
        o = B1 + 3 * c
        return wp.vec3(m[a, o], m[a, o + 1], m[a, o + 2])

    @wp.func
    def mat_c(m: wp.array2d(dtype=wp.float32), a: int, c: int):
        # unique order (xx, xy, xz, yy, yz, zz)
        o = B2 + 6 * c
        xx = m[a, o]
        xy = m[a, o + 1]
        xz = m[a, o + 2]
        yy = m[a, o + 3]
        yz = m[a, o + 4]
        zz = m[a, o + 5]
        return wp.mat33(xx, xy, xz, xy, yy, yz, xz, yz, zz)

    @wp.func
    def mats_d(m: wp.array2d(dtype=wp.float32), a: int, c: int):
        # unique order (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz); D[i]_jk = d_ijk
        o = B3 + 10 * c
        q0 = m[a, o]
        q1 = m[a, o + 1]
        q2 = m[a, o + 2]
        q3 = m[a, o + 3]
        q4 = m[a, o + 4]
        q5 = m[a, o + 5]
        q6 = m[a, o + 6]
        q7 = m[a, o + 7]
        q8 = m[a, o + 8]
        q9 = m[a, o + 9]
        d0 = wp.mat33(q0, q1, q2, q1, q3, q4, q2, q4, q5)
        d1 = wp.mat33(q1, q3, q4, q3, q6, q7, q4, q7, q8)
        d2 = wp.mat33(q2, q4, q5, q4, q7, q8, q5, q8, q9)
        return d0, d1, d2

    @wp.func
    def row(A: wp.mat33, i: int):
        return wp.vec3(A[i, 0], A[i, 1], A[i, 2])

    @wp.kernel(module="unique")
    def gm_contract_fwd(
        m: wp.array2d(dtype=wp.float32),
        table: wp.array2d(dtype=wp.int32),
        feats: wp.array2d(dtype=wp.float32),
    ):
        a, f = wp.tid()
        fam = table[f, 0]
        c0 = table[f, 1]
        c1 = table[f, 2]
        c2 = table[f, 3]
        v = wp.float32(0.0)
        if fam == 1:
            v = wp.dot(vec_b(m, a, c0), vec_b(m, a, c1))
        elif fam == 2:
            v = wp.ddot(mat_c(m, a, c0), mat_c(m, a, c1))
        elif fam == 3:
            x0, x1, x2 = mats_d(m, a, c0)
            y0, y1, y2 = mats_d(m, a, c1)
            v = wp.ddot(x0, y0) + wp.ddot(x1, y1) + wp.ddot(x2, y2)
        elif fam == 4:
            v = wp.ddot(mat_c(m, a, c0), mat_c(m, a, c1) @ wp.transpose(mat_c(m, a, c2)))
        elif fam == 5:
            v = wp.dot(vec_b(m, a, c0), mat_c(m, a, c2) @ vec_b(m, a, c1))
        elif fam == 6:
            x0, x1, x2 = mats_d(m, a, c0)
            y0, y1, y2 = mats_d(m, a, c1)
            C2 = mat_c(m, a, c2)
            v = wp.ddot(x0 @ C2, y0) + wp.ddot(x1 @ C2, y1) + wp.ddot(x2 @ C2, y2)
        elif fam == 7:
            x0, x1, x2 = mats_d(m, a, c0)
            C1 = mat_c(m, a, c1)
            b2 = vec_b(m, a, c2)
            v = (
                wp.dot(row(C1, 0), x0 @ b2)
                + wp.dot(row(C1, 1), x1 @ b2)
                + wp.dot(row(C1, 2), x2 @ b2)
            )
        feats[a, f] = v

    @wp.func
    def vec_bT(m: wp.array2d(dtype=wp.float32), a: int, c: int):
        o = B1 + 3 * c
        return wp.vec3(m[o, a], m[o + 1, a], m[o + 2, a])

    @wp.func
    def mat_cT(m: wp.array2d(dtype=wp.float32), a: int, c: int):
        o = B2 + 6 * c
        xx = m[o, a]
        xy = m[o + 1, a]
        xz = m[o + 2, a]
        yy = m[o + 3, a]
        yz = m[o + 4, a]
        zz = m[o + 5, a]
        return wp.mat33(xx, xy, xz, xy, yy, yz, xz, yz, zz)

    @wp.func
    def mats_dT(m: wp.array2d(dtype=wp.float32), a: int, c: int):
        o = B3 + 10 * c
        q0 = m[o, a]
        q1 = m[o + 1, a]
        q2 = m[o + 2, a]
        q3 = m[o + 3, a]
        q4 = m[o + 4, a]
        q5 = m[o + 5, a]
        q6 = m[o + 6, a]
        q7 = m[o + 7, a]
        q8 = m[o + 8, a]
        q9 = m[o + 9, a]
        d0 = wp.mat33(q0, q1, q2, q1, q3, q4, q2, q4, q5)
        d1 = wp.mat33(q1, q3, q4, q3, q6, q7, q4, q7, q8)
        d2 = wp.mat33(q2, q4, q5, q4, q7, q8, q5, q8, q9)
        return d0, d1, d2

    @wp.func
    def add_b(grad: wp.array2d(dtype=wp.float32), a: int, c: int, dB: wp.vec3):
        o = B1 + 3 * c
        for i in range(3):
            wp.atomic_add(grad, o + i, a, dB[i])

    @wp.func
    def add_c(grad: wp.array2d(dtype=wp.float32), a: int, c: int, dC: wp.mat33):
        # full-tensor gradient -> packed unique components (sum over equivalent entries)
        o = B2 + 6 * c
        wp.atomic_add(grad, o, a, dC[0, 0])
        wp.atomic_add(grad, o + 1, a, dC[0, 1] + dC[1, 0])
        wp.atomic_add(grad, o + 2, a, dC[0, 2] + dC[2, 0])
        wp.atomic_add(grad, o + 3, a, dC[1, 1])
        wp.atomic_add(grad, o + 4, a, dC[1, 2] + dC[2, 1])
        wp.atomic_add(grad, o + 5, a, dC[2, 2])

    @wp.func
    def add_d(
        grad: wp.array2d(dtype=wp.float32),
        a: int,
        c: int,
        d0: wp.mat33,
        d1: wp.mat33,
        d2: wp.mat33,
    ):
        o = B3 + 10 * c
        wp.atomic_add(grad, o, a, d0[0, 0])
        wp.atomic_add(grad, o + 1, a, d0[0, 1] + d0[1, 0] + d1[0, 0])
        wp.atomic_add(grad, o + 2, a, d0[0, 2] + d0[2, 0] + d2[0, 0])
        wp.atomic_add(grad, o + 3, a, d0[1, 1] + d1[0, 1] + d1[1, 0])
        wp.atomic_add(
            grad,
            o + 4,
            a,
            d0[1, 2] + d0[2, 1] + d1[0, 2] + d1[2, 0] + d2[0, 1] + d2[1, 0],
        )
        wp.atomic_add(grad, o + 5, a, d0[2, 2] + d2[0, 2] + d2[2, 0])
        wp.atomic_add(grad, o + 6, a, d1[1, 1])
        wp.atomic_add(grad, o + 7, a, d1[1, 2] + d1[2, 1] + d2[1, 1])
        wp.atomic_add(grad, o + 8, a, d1[2, 2] + d2[1, 2] + d2[2, 1])
        wp.atomic_add(grad, o + 9, a, d2[2, 2])

    def make_bwd(FAM: int, POS: int = -1):
        """Backward kernel specialised to one family, and for POS >= 0 to one slot position
        (branches resolved at codegen, so only those temporaries are live). Families 6 and 7
        are split per position: in one kernel their three branches' D-tensor temporaries
        needed 126-166 registers (16-31 % occupancy, E25)."""
        KIND = _KIND.get((FAM, POS), -1)

        @wp.kernel(module="unique")
        def gm_contract_bwd(
            m: wp.array2d(dtype=wp.float32),
            table: wp.array2d(dtype=wp.int32),
            chunks: wp.array2d(dtype=wp.int32),
            entries: wp.array(dtype=wp.int32),
            g: wp.array2d(dtype=wp.float32),
            grad: wp.array2d(dtype=wp.float32),  # in/out, zero-initialised
        ):
            # thread (chunk, atom), atoms fastest: all lanes of a warp run the same chunk, i.e. the
            # same (channel r, family, slot position) and the same features -> no divergence
            ci, a = wp.tid()
            r = chunks[ci, 0]
            pos = chunks[ci, 2]
            start = chunks[ci, 3]
            count = chunks[ci, 4]
            dB = wp.vec3()
            dC = wp.mat33()
            dD0 = wp.mat33()
            dD1 = wp.mat33()
            dD2 = wp.mat33()
            for e in range(count):
                f = entries[start + e]
                c0 = table[f, 1]
                c1 = table[f, 2]
                c2 = table[f, 3]
                gf = g[f, a]
                if wp.static(FAM == 1):
                    if pos == 0:
                        dB += gf * vec_bT(m, a, c1)
                    else:
                        dB += gf * vec_bT(m, a, c0)
                if wp.static(FAM == 2):
                    if pos == 0:
                        dC += gf * mat_cT(m, a, c1)
                    else:
                        dC += gf * mat_cT(m, a, c0)
                if wp.static(FAM == 3):
                    o = c1
                    if pos == 1:
                        o = c0
                    y0, y1, y2 = mats_dT(m, a, o)
                    dD0 += gf * y0
                    dD1 += gf * y1
                    dD2 += gf * y2
                if wp.static(FAM == 4):
                    if pos == 0:
                        dC += gf * (mat_cT(m, a, c1) @ wp.transpose(mat_cT(m, a, c2)))
                    elif pos == 1:
                        dC += gf * (mat_cT(m, a, c0) @ mat_cT(m, a, c2))
                    else:
                        dC += gf * (wp.transpose(mat_cT(m, a, c0)) @ mat_cT(m, a, c1))
                if wp.static(FAM == 5):
                    if pos == 0:
                        dB += gf * (mat_cT(m, a, c2) @ vec_bT(m, a, c1))
                    elif pos == 1:
                        dB += gf * (wp.transpose(mat_cT(m, a, c2)) @ vec_bT(m, a, c0))
                    else:
                        dC += gf * wp.outer(vec_bT(m, a, c0), vec_bT(m, a, c1))
                if wp.static(FAM == 6):
                    C2 = mat_cT(m, a, c2)
                    if wp.static(POS == 0):
                        y0, y1, y2 = mats_dT(m, a, c1)
                        C2t = wp.transpose(C2)
                        dD0 += gf * (y0 @ C2t)
                        dD1 += gf * (y1 @ C2t)
                        dD2 += gf * (y2 @ C2t)
                    if wp.static(POS == 1):
                        x0, x1, x2 = mats_dT(m, a, c0)
                        dD0 += gf * (x0 @ C2)
                        dD1 += gf * (x1 @ C2)
                        dD2 += gf * (x2 @ C2)
                    if wp.static(POS == 2):
                        x0, x1, x2 = mats_dT(m, a, c0)
                        y0, y1, y2 = mats_dT(m, a, c1)
                        dC += gf * (
                            wp.transpose(x0) @ y0
                            + wp.transpose(x1) @ y1
                            + wp.transpose(x2) @ y2
                        )
                if wp.static(FAM == 7):
                    if wp.static(POS == 0):
                        C1 = mat_cT(m, a, c1)
                        b2 = vec_bT(m, a, c2)
                        dD0 += gf * wp.outer(row(C1, 0), b2)
                        dD1 += gf * wp.outer(row(C1, 1), b2)
                        dD2 += gf * wp.outer(row(C1, 2), b2)
                    if wp.static(POS == 1):
                        x0, x1, x2 = mats_dT(m, a, c0)
                        b2 = vec_bT(m, a, c2)
                        v0 = x0 @ b2
                        v1 = x1 @ b2
                        v2 = x2 @ b2
                        dC += gf * wp.mat33(
                            v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]
                        )
                    if wp.static(POS == 2):
                        x0, x1, x2 = mats_dT(m, a, c0)
                        C1 = mat_cT(m, a, c1)
                        dB += gf * (
                            wp.transpose(x0) @ row(C1, 0)
                            + wp.transpose(x1) @ row(C1, 1)
                            + wp.transpose(x2) @ row(C1, 2)
                        )
            if wp.static(KIND == 0):
                add_b(grad, a, r, dB)
            if wp.static(KIND == 1):
                add_c(grad, a, r, dC)
            if wp.static(KIND == 2):
                add_d(grad, a, r, dD0, dD1, dD2)
            if wp.static(KIND < 0):
                kind = chunks[
                    ci, 5
                ]  # which tensor of channel r this chunk differentiates
                if kind == 0:
                    add_b(grad, a, r, dB)
                elif kind == 1:
                    add_c(grad, a, r, dC)
                else:
                    add_d(grad, a, r, dD0, dD1, dD2)

        return wp.jax_kernel(gm_contract_bwd, num_outputs=1, in_out_argnames=["grad"])

    fwd = wp.jax_kernel(gm_contract_fwd, num_outputs=1)
    # one kernel per family; families 6 and 7 one per (family, slot position)
    bwd = {(fam, -1): make_bwd(fam) for fam in range(1, 6)}
    bwd.update({(fam, p): make_bwd(fam, p) for fam in (6, 7) for p in range(3)})
    return fwd, bwd


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _contract(cfg, packed):
    K, C, table = cfg
    fwd, _ = _kernels(K, C)
    n_atoms, n_feat = packed.shape[0], len(table)
    (feats,) = fwd(
        packed,
        jnp.asarray(np.asarray(table, np.int32)),
        launch_dims=(n_atoms, n_feat),
        output_dims=(n_atoms, n_feat),
    )
    return feats


def _contract_fwd(cfg, packed):
    return _contract(cfg, packed), packed


# tensor differentiated by (family, slot position): 0 = b (m1), 1 = C (m2), 2 = D (m3)
_KIND = {
    (1, 0): 0,
    (1, 1): 0,
    (2, 0): 1,
    (2, 1): 1,
    (3, 0): 2,
    (3, 1): 2,
    (4, 0): 1,
    (4, 1): 1,
    (4, 2): 1,
    (5, 0): 0,
    (5, 1): 0,
    (5, 2): 1,
    (6, 0): 2,
    (6, 1): 2,
    (6, 2): 1,
    (7, 0): 2,
    (7, 1): 1,
    (7, 2): 0,
}
_N_SLOTS = {1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3}


def chunk_length(n_atoms):
    """Features per backward thread: short chunks give small systems enough threads (each
    (family, position) kernel launches n_chunks x N threads), long ones save atomics at large
    N. L40S (E25): 3.9k best at 4 (2 / 4 / 8 / 16 / 32: 2140 / 2154 / 2110 / 2029 / 1940
    steps/s), 17.9k flat 4-64, 104k 32 > 4 by 1.7 %."""
    return int(min(32, max(4, 2 ** round(math.log2(max(1.0, n_atoms / 1000.0))))))


@functools.lru_cache(maxsize=None)
def backward_chunks(table, chunk=32):
    """Static backward work list: every (feature f, slot position p) occurrence, grouped by
    (channel table[f, 1 + p], family, p) and split into chunks of <= `chunk` features.
    Returns chunks (n_chunks, 6) int32 rows (channel, family, position, start, count, kind)
    and entries (n_occurrences,) int32 feature ids."""
    groups = {}
    for f, (fam, *cs) in enumerate(table):
        for p in range(_N_SLOTS[fam]):
            groups.setdefault((cs[p], fam, p), []).append(f)
    chunks, entries = [], []
    for (c, fam, p), fs in sorted(groups.items()):
        for k in range(0, len(fs), chunk):
            part = fs[k : k + chunk]
            chunks.append((c, fam, p, len(entries), len(part), _KIND[(fam, p)]))
            entries += part
    return np.asarray(chunks, np.int32), np.asarray(entries, np.int32)


def _contract_bwd(cfg, packed, g):
    K, C, table = cfg
    _, bwd = _kernels(K, C)
    chunks, entries = backward_chunks(table, chunk_length(packed.shape[0]))
    table_a, entries_a = jnp.asarray(np.asarray(table, np.int32)), jnp.asarray(entries)
    # component-major copies: a warp's lanes are consecutive atoms, so every load and atomic
    # of the backward kernels is one coalesced line instead of 32 scattered ones
    mT, gT = packed.T, g.astype(jnp.float32).T
    gradT = jnp.zeros_like(mT)
    for fam, p in sorted({(f, p if f >= 6 else -1) for f, p in chunks[:, 1:3].tolist()}):
        sel = (chunks[:, 1] == fam) & ((chunks[:, 2] == p) if p >= 0 else True)
        ch = chunks[sel]
        (gradT,) = bwd[(fam, p)](
            mT,
            table_a,
            jnp.asarray(ch),
            entries_a,
            gT,
            gradT,
            launch_dims=(len(ch), packed.shape[0]),
        )
    return (gradT.T,)


_contract.defvjp(_contract_fwd, _contract_bwd)


def warp_contract(packed, K, C, n_contr=8, drop_contractions=()):
    """Gaussian-moment features [m0 | contr_1..7] (n_atoms, n_features) from the packed
    moments (n_atoms, K + 19 C) of `warp_moments`, identical in layout to `_contract`."""
    packed = packed.astype(jnp.float32)
    table = tuple(map(tuple, feature_table(C, n_contr, drop_contractions).tolist()))
    parts = []
    if n_contr > 0 and 0 not in drop_contractions:
        parts.append(packed[:, :K])
    if table:
        parts.append(_contract((int(K), int(C), table), packed))
    return jnp.concatenate(parts, axis=-1)
