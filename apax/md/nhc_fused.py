"""jax-md's Nose-Hoover chain with the half step written as straight-line scalar code.

Upstream ``half_step_chain_fn`` runs the chain_steps * sy_steps substeps in a lax.scan
(an XLA while loop), each substep has two more small scans over the chain links and
rescales the whole momentum array. On GPU that is a few hundred tiny sequential kernels
per MD step. Here the substeps and chain links are unrolled in Python (all their sizes
are static), the chain is kept as scalars, and the momentum is rescaled once per half
step by the product of the substep scales. The arithmetic is the same as jax_md 0.2.29
``simulate.nose_hoover_chain``; only the order of the momentum multiplications differs.
"""
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax_md import dataclasses, util
from jax_md.simulate import (
    SUZUKI_YOSHIDA_WEIGHTS,
    NoseHooverChain,
    NoseHooverChainFns,
)

f32 = util.f32


def nose_hoover_chain_fused(
    dt: float, chain_length: int, chain_steps: int, sy_steps: int, tau: float
) -> NoseHooverChainFns:
    def init_fn(degrees_of_freedom, KE, kT):
        xi = jnp.zeros(chain_length, KE.dtype)
        p_xi = jnp.zeros(chain_length, KE.dtype)

        Q = kT * tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
        Q = Q.at[0].multiply(degrees_of_freedom)
        return NoseHooverChain(xi, p_xi, Q, tau, KE, degrees_of_freedom)

    def substep(delta, xi, p_xi, Q, KE, DOF, kT):
        """One chain update on lists of scalars; returns the momentum scale factor."""
        delta_2 = delta / f32(2.0)
        delta_4 = delta_2 / f32(2.0)
        delta_8 = delta_4 / f32(2.0)

        M = chain_length - 1
        p_xi = list(p_xi)

        G = p_xi[M - 1] ** f32(2) / Q[M - 1] - kT
        p_xi[M] = p_xi[M] + delta_4 * G

        # backward sweep m = M-1 .. 1; each update uses the old p_xi[m - 1], p_xi[m]
        # and the newly updated p_xi[m + 1]
        p_new = p_xi[M]
        for m in range(M - 1, 0, -1):
            G = p_xi[m - 1] ** 2 / Q[m - 1] - kT
            scale = jnp.exp(-delta_8 * p_new / Q[m + 1])
            p_new = scale * (scale * p_xi[m] + delta_4 * G)
            p_xi[m] = p_new

        G = f32(2.0) * KE - DOF * kT
        scale = jnp.exp(-delta_8 * p_xi[1] / Q[1])
        p_xi[0] = scale * (scale * p_xi[0] + delta_4 * G)

        p_scale = jnp.exp(-delta_2 * p_xi[0] / Q[0])
        KE = KE * p_scale ** f32(2)

        xi = [xi[m] + delta_2 * p_xi[m] / Q[m] for m in range(chain_length)]

        G = f32(2) * KE - DOF * kT
        # forward sweep m = 0 .. M-1; uses the p_xi[m + 1] from before this sweep
        p_old = list(p_xi)
        for m in range(M):
            scale = jnp.exp(-delta_8 * p_old[m + 1] / Q[m + 1])
            p_xi[m] = scale * (scale * p_old[m] + delta_4 * G)
            G = p_xi[m] ** 2 / Q[m] - kT
        p_xi[M] = p_xi[M] + delta_4 * G

        return xi, p_xi, KE, p_scale

    def half_step_chain_fn(P, state, kT):
        xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)
        xi = [xi[m] for m in range(chain_length)]
        p_xi = [p_xi[m] for m in range(chain_length)]
        Qs = [Q[m] for m in range(chain_length)]

        if chain_steps == 1 and sy_steps == 1:
            deltas = [dt]
        else:
            ws = SUZUKI_YOSHIDA_WEIGHTS[sy_steps]
            deltas = [
                f32(dt / chain_steps * ws[i % sy_steps])
                for i in range(chain_steps * sy_steps)
            ]

        total_scale = 1.0
        for delta in deltas:
            xi, p_xi, KE, s = substep(delta, xi, p_xi, Qs, KE, DOF, kT)
            total_scale = total_scale * s

        P = tree_map(lambda p: p * total_scale, P)
        return P, NoseHooverChain(jnp.stack(xi), jnp.stack(p_xi), Q, _tau, KE, DOF)

    def update_chain_mass_fn(state, kT):
        xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)

        Q = kT * _tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
        Q = Q.at[0].multiply(DOF)

        return NoseHooverChain(xi, p_xi, Q, _tau, KE, DOF)

    return NoseHooverChainFns(init_fn, half_step_chain_fn, update_chain_mass_fn)
