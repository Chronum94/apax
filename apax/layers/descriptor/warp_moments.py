"""Fused Gaussian-moment kernels (NVIDIA Warp) for inference.

Replaces, for a tabulated radial function (`TabulatedRadialFunction` with the Bessel
basis and the identity radial transform), the per-pair pipeline of
`GaussianMomentDescriptor`

    distance -> basis -> per-pair-type coefficients -> cutoff -> mask
    -> radial compression -> packed direction products -> 4 x segment_sum

by one forward kernel and one backward kernel. No per-pair intermediate is written to
global memory and neither kernel uses atomics:

* forward, atom-centric: one thread per (centre atom, tensor channel c, split) walks a strided
  share of the atom's contiguous neighbour segment (jax-md sparse lists are sorted by centre),
  evaluates the Bessel basis once per pair and accumulates the packed rank-1/2/3 moments of
  tensor channel c and the scalar channels c, c + C, ... in registers (atomics only across the
  split);
* backward, pair-parallel: one thread per pair reads the moment cotangents of its
  centre atom and writes dE/d(dr_vec) of that pair. Gradients w.r.t. positions,
  forces and the virial then follow from JAX's own gather backward in
  `compute_distances`.

The radial compression (n_radial -> n_radial_tensor, linear, no bias) is folded into a
second per-pair-type table, so the result equals the XLA path up to fp32 rounding.
Inference only: parameters get no gradient, and second derivatives are not supported.

Output layout per atom (float32): [m0 (K) | m1 (C, 3) | p2 (C, 6) | p3 (C, 10)], with the
packed component order of `moments.geometric_moments_packed`.
"""

import functools
import math

import jax
import jax.numpy as jnp

_EPS_DN = 1e-5  # dn = dr_vec / (|dr_vec| + 1e-5), as in GaussianMomentDescriptor


@functools.lru_cache(maxsize=None)
def _kernels(
    n_basis: int,
    n_radial: int,
    n_tensor: int,
    r_max: float,
    n_split: int = 1,
    nlh_rmax: float = 0.0,
    nlh_scale: float = 14.4,
):
    """Build (and cache) the Warp kernels for one descriptor shape; sizes and r_max are
    compile-time constants, so all channel loops unroll. n_split threads share one
    (atom, channel): they take every n_split-th pair of the atom's segment and combine
    their partial sums with atomic adds (more parallelism for small systems)."""
    import warp as wp

    NB, K, C, S = n_basis, n_radial, n_tensor, n_split
    F = K + 19 * C
    # optional fused NLH repulsion (apax.layers.empirical.NLHRepulsion): per-atom energy in
    # output column F, its gradient added to dE/d(dr_vec) in the backward kernel
    NLH = 1 if nlh_rmax > 0.0 else 0
    RN = float(nlh_rmax) if NLH else 1.0
    PI_RN = math.pi / RN
    HALF_SCALE = 0.5 * float(nlh_scale)
    F_OUT = F
    KPER = (K + C - 1) // C  # scalar channels per thread
    if KPER > 4:
        raise ValueError(f"warp moments: n_radial ({K}) > 4 x n_radial_tensor ({C}) not supported")
    R = float(r_max)
    PI_R = math.pi / R
    A = math.sqrt(2.0) * math.pi / R**1.5  # Bessel prefactor; the sign alternates with n
    vecB = wp.types.vector(length=NB, dtype=wp.float32)

    @wp.func
    def bessel(r: wp.float32):
        # b_n(r) = A (-1)^n [sinc_{n+1} + sinc_{n+2}],  sinc_m = sin(m t) / (m t),  t = pi r / R
        # db_n/dr; sin/cos(m t) by the Chebyshev recurrence from one sincos
        th = wp.max(r * wp.float32(PI_R), wp.float32(1e-6))
        inv = wp.float32(1.0) / th
        c1 = wp.cos(th)
        s_prev = wp.float32(0.0)
        s_cur = wp.sin(th)
        c_prev = wp.float32(1.0)
        c_cur = c1
        f_cur = s_cur * inv
        d_cur = wp.float32(PI_R) * (c_cur * inv - s_cur * inv * inv)
        b = vecB()
        db = vecB()
        sign = wp.float32(A)
        for n in range(NB):
            s_next = wp.float32(2.0) * c1 * s_cur - s_prev
            c_next = wp.float32(2.0) * c1 * c_cur - c_prev
            inv_m = wp.float32(1.0) / wp.float32(n + 2)
            f_next = s_next * inv * inv_m
            d_next = wp.float32(PI_R) * (c_next * inv - s_next * inv * inv * inv_m)
            b[n] = sign * (f_cur + f_next)
            db[n] = sign * (d_cur + d_next)
            sign = -sign
            s_prev = s_cur
            s_cur = s_next
            c_prev = c_cur
            c_cur = c_next
            f_cur = f_next
            d_cur = d_next
        return b, db

    @wp.func
    def cutoff(r: wp.float32):
        rc = wp.min(r, wp.float32(R))
        fc = wp.float32(0.5) * (wp.cos(wp.float32(PI_R) * rc) + wp.float32(1.0))
        dfc = wp.float32(0.0)
        if r < wp.float32(R):
            dfc = wp.float32(-0.5 * PI_R) * wp.sin(wp.float32(PI_R) * rc)
        return fc, dfc

    @wp.kernel(module="unique")
    def gm_forward(
        starts: wp.array(dtype=wp.int32),
        pairs: wp.array2d(dtype=wp.int32),  # row 0 neighbour, row 1 centre
        Z: wp.array(dtype=wp.int32),
        species_map: wp.array(dtype=wp.int32),
        dr_vec: wp.array(dtype=wp.vec3),
        T0: wp.array3d(dtype=wp.float32),
        T1: wp.array3d(dtype=wp.float32),
        n_el: wp.int32,
        out: wp.array2d(dtype=wp.float32),  # in/out, zero-initialised
    ):
        # thread (atom a, tensor channel k < C, split s): tensor moments of channel k and the
        # scalar channels k, k + C, k + 2C, ... < K; the basis is evaluated once per pair per
        # thread (C threads per atom-split instead of K)
        a, k, s = wp.tid()
        zc = species_map[Z[a]] * n_el
        acc0 = wp.vec4()  # scalar channels k + i*C, i < KPER (KPER <= 4)
        m1 = wp.vec3()
        q2a = wp.vec3()  # xx, xy, xz
        q2b = wp.vec3()  # yy, yz, zz
        q3a = wp.vec4()  # xxx, xxy, xxz, xyy
        q3b = wp.vec3()  # xyz, xzz, yyy
        q3c = wp.vec3()  # yyz, yzz, zzz
        for p in range(starts[a] + s, starts[a + 1], S):
            j = pairs[0, p]
            v = dr_vec[p]
            r2 = wp.dot(v, v)
            if j == a and r2 <= wp.float32(1e-12):
                continue  # padding (masked as in mask_by_neighbor)
            r = wp.float32(0.0)
            if r2 > wp.float32(0.0):
                r = wp.sqrt(r2)
            t = zc + species_map[Z[j]]
            b, db = bessel(r)
            fc, dfc = cutoff(r)
            for i in range(KPER):
                kk = k + i * C
                if kk < K:
                    rk = wp.float32(0.0)
                    for n in range(NB):
                        rk += b[n] * T0[t, n, kk]
                    acc0[i] = acc0[i] + rk * fc
            r1 = wp.float32(0.0)
            for n in range(NB):
                r1 += b[n] * T1[t, n, k]
            r1 = r1 * fc
            d = v / (r + wp.float32(_EPS_DN))
            x = d[0]
            y = d[1]
            z = d[2]
            m1 += r1 * d
            q2a += r1 * wp.vec3(x * x, x * y, x * z)
            q2b += r1 * wp.vec3(y * y, y * z, z * z)
            q3a += r1 * wp.vec4(x * x * x, x * x * y, x * x * z, x * y * y)
            q3b += r1 * wp.vec3(x * y * z, x * z * z, y * y * y)
            q3c += r1 * wp.vec3(y * y * z, y * z * z, z * z * z)
        for i in range(KPER):
            kk = k + i * C
            if kk < K:
                wp.atomic_add(out, a, kk, acc0[i])
        o1 = K + 3 * k
        o2 = K + 3 * C + 6 * k
        o3 = K + 9 * C + 10 * k
        for i in range(3):
            wp.atomic_add(out, a, o1 + i, m1[i])
            wp.atomic_add(out, a, o2 + i, q2a[i])
            wp.atomic_add(out, a, o2 + 3 + i, q2b[i])
            wp.atomic_add(out, a, o3 + 4 + i, q3b[i])
            wp.atomic_add(out, a, o3 + 7 + i, q3c[i])
        for i in range(4):
            wp.atomic_add(out, a, o3 + i, q3a[i])

    @wp.kernel(module="unique")
    def nlh_energy(
        starts: wp.array(dtype=wp.int32),
        pairs: wp.array2d(dtype=wp.int32),
        Z: wp.array(dtype=wp.int32),
        dr_vec: wp.array(dtype=wp.vec3),
        nlh_a: wp.array3d(dtype=wp.float32),
        nlh_b: wp.array3d(dtype=wp.float32),
        e: wp.array(dtype=wp.float32),  # in/out, zero-initialised
    ):
        # per-atom NLH energy: 0.5 * sum over the atom's pairs of
        # 14.4 Z_i Z_j / r * sum_m a exp(-b r) * cosine cutoff, r clipped to [0.02, RN].
        # A separate kernel (not folded into gm_forward, where only one channel-thread per
        # warp would do it: divergence) so that XLA drops it when the energy is unused.
        a, s = wp.tid()
        acc = wp.float32(0.0)
        zj = Z[a]
        for p in range(starts[a] + s, starts[a + 1], S):
            j = pairs[0, p]
            v = dr_vec[p]
            r2 = wp.dot(v, v)
            if j == a and r2 <= wp.float32(1e-12):
                continue
            r = wp.float32(0.0)
            if r2 > wp.float32(0.0):
                r = wp.sqrt(r2)
            rn = wp.clamp(r, wp.float32(0.02), wp.float32(RN))
            zi = Z[j]
            phi = wp.float32(0.0)
            for m in range(3):
                phi += nlh_a[zi, zj, m] * wp.exp(-nlh_b[zi, zj, m] * rn)
            fcn = wp.float32(0.5) * (wp.cos(wp.float32(PI_RN) * rn) + wp.float32(1.0))
            acc += wp.float32(HALF_SCALE) * wp.float32(zi * zj) / rn * phi * fcn
        wp.atomic_add(e, a, acc)

    @wp.kernel(module="unique")
    def m0_cotangent_table(
        Z: wp.array(dtype=wp.int32),
        species_map: wp.array(dtype=wp.int32),
        T0: wp.array3d(dtype=wp.float32),
        g: wp.array2d(dtype=wp.float32),
        n_el: wp.int32,
        G0: wp.array3d(dtype=wp.float32),
    ):
        # G0[a, e, n] = sum_k T0[type(a, e), n, k] g[a, k]: the m0 part of the backward depends
        # only on (centre atom, neighbour element), not on the pair
        a, e, n = wp.tid()
        t = species_map[Z[a]] * n_el + e
        acc = wp.float32(0.0)
        for k in range(K):
            acc += T0[t, n, k] * g[a, k]
        G0[a, e, n] = acc

    @wp.kernel(module="unique")
    def gm_backward(
        pairs: wp.array2d(dtype=wp.int32),
        Z: wp.array(dtype=wp.int32),
        species_map: wp.array(dtype=wp.int32),
        dr_vec: wp.array(dtype=wp.vec3),
        T0: wp.array3d(dtype=wp.float32),
        T1: wp.array3d(dtype=wp.float32),
        nlh_a: wp.array3d(dtype=wp.float32),
        nlh_b: wp.array3d(dtype=wp.float32),
        g: wp.array2d(dtype=wp.float32),
        G0: wp.array3d(dtype=wp.float32),
        g_nlh: wp.array(dtype=wp.float32),
        n_atoms: wp.int32,
        n_el: wp.int32,
        grad: wp.array(dtype=wp.vec3),
    ):
        p = wp.tid()
        a = pairs[1, p]
        j = pairs[0, p]
        v = dr_vec[p]
        r2 = wp.dot(v, v)
        if a >= n_atoms or j >= n_atoms or (j == a and r2 <= wp.float32(1e-12)) or r2 <= wp.float32(0.0):
            grad[p] = wp.vec3()
            return
        r = wp.sqrt(r2)
        ej = species_map[Z[j]]
        t = species_map[Z[a]] * n_el + ej
        b, db = bessel(r)
        fc, dfc = cutoff(r)
        # w_n = b_n fc (radial basis incl. cutoff), q_n = d(w_n)/dr; m0 part via the G0 table
        dEdr = wp.float32(0.0)
        for n in range(NB):
            dEdr += (db[n] * fc + b[n] * dfc) * G0[a, ej, n]
        if NLH == 1:
            # d/dr of the NLH pair energy (zero outside the clip range, as jnp.clip)
            if r > wp.float32(0.02) and r < wp.float32(RN):
                zi = Z[j]
                zj = Z[a]
                phi = wp.float32(0.0)
                dphi = wp.float32(0.0)
                for m in range(3):
                    am = nlh_a[zi, zj, m]
                    bm = nlh_b[zi, zj, m]
                    em = wp.exp(-bm * r)
                    phi += am * em
                    dphi -= bm * am * em
                fcn = wp.float32(0.5) * (wp.cos(wp.float32(PI_RN) * r) + wp.float32(1.0))
                dfcn = wp.float32(-0.5 * PI_RN) * wp.sin(wp.float32(PI_RN) * r)
                c = wp.float32(HALF_SCALE) * wp.float32(zi * zj)
                dEdr += g_nlh[a] * c * ((dphi * fcn + phi * dfcn) / r - phi * fcn / (r * r))
        rp = r + wp.float32(_EPS_DN)
        d = v / rp
        x = d[0]
        y = d[1]
        z = d[2]
        h = wp.vec3()  # dE/d(dn)
        for c in range(C):
            r1 = wp.float32(0.0)
            dr1 = wp.float32(0.0)
            for n in range(NB):
                r1 += b[n] * T1[t, n, c]
                dr1 += db[n] * T1[t, n, c]
            dr1 = dr1 * fc + r1 * dfc
            r1 = r1 * fc
            o1 = K + 3 * c
            o2 = K + 3 * C + 6 * c
            o3 = K + 9 * C + 10 * c
            g1 = wp.vec3(g[a, o1], g[a, o1 + 1], g[a, o1 + 2])
            # packed rank-2: xx xy xz yy yz zz
            g00 = g[a, o2]
            g01 = g[a, o2 + 1]
            g02 = g[a, o2 + 2]
            g11 = g[a, o2 + 3]
            g12 = g[a, o2 + 4]
            g22 = g[a, o2 + 5]
            # packed rank-3: xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
            h000 = g[a, o3]
            h001 = g[a, o3 + 1]
            h002 = g[a, o3 + 2]
            h011 = g[a, o3 + 3]
            h012 = g[a, o3 + 4]
            h022 = g[a, o3 + 5]
            h111 = g[a, o3 + 6]
            h112 = g[a, o3 + 7]
            h122 = g[a, o3 + 8]
            h222 = g[a, o3 + 9]
            S = wp.dot(g1, d)
            S += g00 * x * x + g01 * x * y + g02 * x * z + g11 * y * y + g12 * y * z + g22 * z * z
            S += (
                h000 * x * x * x
                + h001 * x * x * y
                + h002 * x * x * z
                + h011 * x * y * y
                + h012 * x * y * z
                + h022 * x * z * z
                + h111 * y * y * y
                + h112 * y * y * z
                + h122 * y * z * z
                + h222 * z * z * z
            )
            gx = g1[0] + wp.float32(2.0) * g00 * x + g01 * y + g02 * z
            gy = g1[1] + g01 * x + wp.float32(2.0) * g11 * y + g12 * z
            gz = g1[2] + g02 * x + g12 * y + wp.float32(2.0) * g22 * z
            gx += (
                wp.float32(3.0) * h000 * x * x
                + wp.float32(2.0) * h001 * x * y
                + wp.float32(2.0) * h002 * x * z
                + h011 * y * y
                + h012 * y * z
                + h022 * z * z
            )
            gy += (
                h001 * x * x
                + wp.float32(2.0) * h011 * x * y
                + h012 * x * z
                + wp.float32(3.0) * h111 * y * y
                + wp.float32(2.0) * h112 * y * z
                + h122 * z * z
            )
            gz += (
                h002 * x * x
                + h012 * x * y
                + wp.float32(2.0) * h022 * x * z
                + h112 * y * y
                + wp.float32(2.0) * h122 * y * z
                + wp.float32(3.0) * h222 * z * z
            )
            dEdr += dr1 * S
            h += r1 * wp.vec3(gx, gy, gz)
        # dn = v / (r + eps): d(dn)/dv = I / (r + eps) - v v^T / (r (r + eps)^2)
        grad[p] = (dEdr / r) * v + h / rp - v * (wp.dot(v, h) / (r * rp * rp))

    fwd = wp.jax_kernel(gm_forward, num_outputs=1, in_out_argnames=["out"])
    nlh_fwd = wp.jax_kernel(nlh_energy, num_outputs=1, in_out_argnames=["e"])
    g0_table = wp.jax_kernel(m0_cotangent_table, num_outputs=1)
    bwd = wp.jax_kernel(gm_backward, num_outputs=1)
    return fwd, bwd, F_OUT, nlh_fwd, g0_table


def default_split(n_atoms: int, n_radial: int, scale: float = 3.0e5) -> int:
    """Neighbour-loop split S of the forward kernel (power of two, 1..16); n_radial: threads
    per atom-split (the number of tensor channels). S ~ sqrt(scale / (n_atoms * n_radial)):
    the measured optimum thread count grows ~sqrt(N) (L40S, E24: best S = 4 at 3.9k atoms,
    2 at 17.9k, flat 1-2 at 104k; a fixed thread target cannot match both small sizes)."""
    s = 2 ** round(math.log2(math.sqrt(scale / max(1, n_atoms * n_radial))))
    return int(min(16, max(1, s)))


def warp_packed_moments(
    dr_vec, Z, idx, species_map, T0, T1, r_max, n_split=None, nlh=None, packed=False
):
    """Packed Gaussian moments (atom-major) of a jax-md sparse neighbour list.

    dr_vec: (P, 3) pair vectors as in GaussianMomentDescriptor; Z: (N,) atomic numbers;
    idx: (2, P) with idx[1] the centre (sorted, padding = N at the end);
    species_map: (119,) atomic number -> compact element index;
    T0: (n_types, n_basis, K) radial table; T1: (n_types, n_basis, C) compressed table.
    nlh: optional (a, b, r_max, rep_scale) of an NLHRepulsion to fuse (a, b: (Zmax, Zmax, 3)).
    Returns m0 (N, K), m1 (N, C, 3), p2 (N, C, 6), p3 (N, C, 10) [, e_nlh (N,) per-atom
    NLH energy if nlh is given]; with packed=True the atom-major rows (N, K + 19 C) instead
    of the four blocks (input of `warp_contract`).
    """
    n_atoms = Z.shape[0]
    n_types, n_basis, K = T0.shape
    C = T1.shape[-1]
    n_el = int(round(math.sqrt(n_types)))

    from apax.layers.neighbor_grad import centre_csr

    # contiguous per-centre segments; the stray last slot of jax-md lists (a pair at
    # least the list cutoff apart, zero radial) is turned into padding
    pairs, starts = centre_csr(idx, n_atoms)
    Z = Z.astype(jnp.int32)
    species_map = jnp.asarray(species_map, jnp.int32)
    T0 = jnp.asarray(T0, jnp.float32)
    T1 = jnp.asarray(T1, jnp.float32)

    if n_split is None:
        n_split = default_split(n_atoms, int(C))
    if nlh is not None:
        nlh_a = jnp.asarray(nlh[0], jnp.float32)
        nlh_b = jnp.asarray(nlh[1], jnp.float32)
        nlh_cfg = (float(nlh[2]), float(nlh[3]))
    else:
        nlh_a = nlh_b = jnp.zeros((1, 1, 3), jnp.float32)
        nlh_cfg = (0.0, 14.4)
    cfg = (int(n_basis), int(K), int(C), float(r_max), int(n_atoms), int(n_el), int(n_split)) + nlh_cfg
    out, e_nlh = _moments(
        cfg, dr_vec.astype(jnp.float32), starts, pairs, Z, species_map, T0, T1, nlh_a, nlh_b
    )
    if packed:
        return (out, e_nlh) if nlh is not None else (out,)
    m0 = out[:, :K]
    m1 = out[:, K : K + 3 * C].reshape(n_atoms, C, 3)
    p2 = out[:, K + 3 * C : K + 9 * C].reshape(n_atoms, C, 6)
    p3 = out[:, K + 9 * C : K + 19 * C].reshape(n_atoms, C, 10)
    if nlh is not None:
        return m0, m1, p2, p3, e_nlh
    return m0, m1, p2, p3


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _moments(cfg, dr_vec, starts, pairs, Z, species_map, T0, T1, nlh_a, nlh_b):
    # all arrays are explicit arguments (no closed-over tracers), so this composes with
    # jit / grad / vjp in any order; only dr_vec gets a cotangent
    n_basis, K, C, r_max, n_atoms, n_el, S, nlh_rmax, nlh_scale = cfg
    fwd, _, F, nlh_fwd, _ = _kernels(n_basis, K, C, r_max, S, nlh_rmax, nlh_scale)
    zeros = jnp.zeros((n_atoms, F), jnp.float32)
    (out,) = fwd(
        starts, pairs, Z, species_map, dr_vec, T0, T1, n_el, zeros,
        launch_dims=(n_atoms, C, S),
    )
    if nlh_rmax > 0.0:
        # separate output: XLA removes this call when the energy is not used (force-only steps)
        (e_nlh,) = nlh_fwd(
            starts, pairs, Z, dr_vec, nlh_a, nlh_b, jnp.zeros((n_atoms,), jnp.float32),
            launch_dims=(n_atoms, S),
        )
    else:
        e_nlh = jnp.zeros((n_atoms,), jnp.float32)
    return out, e_nlh


def _moments_fwd(cfg, dr_vec, starts, pairs, Z, species_map, T0, T1, nlh_a, nlh_b):
    out = _moments(cfg, dr_vec, starts, pairs, Z, species_map, T0, T1, nlh_a, nlh_b)
    return out, (dr_vec, pairs, Z, species_map, T0, T1, nlh_a, nlh_b)


def _moments_bwd(cfg, res, cts):
    g, g_nlh = cts
    dr_vec, pairs, Z, species_map, T0, T1, nlh_a, nlh_b = res
    n_basis, K, C, r_max, n_atoms, n_el, S, nlh_rmax, nlh_scale = cfg
    _, bwd, _, _, g0_table = _kernels(n_basis, K, C, r_max, S, nlh_rmax, nlh_scale)
    P = dr_vec.shape[0]
    gf = g.astype(jnp.float32)
    (G0,) = g0_table(
        Z, species_map, T0, gf, n_el,
        launch_dims=(n_atoms, n_el, n_basis), output_dims=(n_atoms, n_el, n_basis),
    )
    (grad,) = bwd(
        pairs, Z, species_map, dr_vec, T0, T1, nlh_a, nlh_b, gf, G0,
        g_nlh.astype(jnp.float32),
        n_atoms, n_el,
        launch_dims=(P,), output_dims=(P, 3),
    )
    return (grad, None, None, None, None, None, None, None, None)


_moments.defvjp(_moments_fwd, _moments_bwd)


def fold_compression(table, W):
    """T1[t] = T0[t] @ W: the radial compression folded into the per-pair-type table."""
    return jnp.einsum("tbk,kc->tbc", table, W, precision=jax.lax.Precision.HIGHEST)


__all__ = ["warp_packed_moments", "fold_compression"]
