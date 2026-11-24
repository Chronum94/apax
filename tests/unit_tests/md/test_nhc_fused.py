"""Straight-line Nose-Hoover chain (md/nhc_fused.py) vs jax-md's lax.scan implementation:
same half step (momentum scale, chain positions / momenta, kinetic energy) to fp64
reassociation round-off, for a chain far from equilibrium (all exp / scale branches active)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import simulate

from apax.md.nhc_fused import nose_hoover_chain_fused


@pytest.mark.parametrize(
    "chain_length, chain_steps, sy_steps", [(3, 2, 3), (5, 1, 1), (4, 2, 5)]
)
def test_fused_chain_matches_jax_md(chain_length, chain_steps, sy_steps):
    with jax.enable_x64(True):
        # dt as an fp32 value, as nvt_nose_hoover_fused passes it (jax-md casts every
        # substep delta to fp32; the fused chain uses dt directly when there is one substep)
        dt, kT, tau = float(np.float32(0.1)), 0.0284, 10.0
        dof = 3 * 500 - 3
        rng = np.random.default_rng(0)
        P = jnp.asarray(rng.normal(size=(500, 3)))
        KE = 0.5 * jnp.sum(P**2)  # unit masses
        ref_fns = simulate.nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, tau)
        new_fns = nose_hoover_chain_fused(dt, chain_length, chain_steps, sy_steps, tau)
        state = ref_fns.initialize(dof, KE, kT)
        state = state.set(
            position=jnp.asarray(rng.normal(size=chain_length)),
            momentum=jnp.asarray(rng.normal(size=chain_length)) * 0.5,
        )
        P_ref, s_ref = ref_fns.half_step(P, state, kT)
        P_new, s_new = new_fns.half_step(P, state, kT)
        np.testing.assert_allclose(P_new, P_ref, rtol=1e-10)
        for field in ("position", "momentum", "kinetic_energy"):
            np.testing.assert_allclose(
                getattr(s_new, field), getattr(s_ref, field), rtol=1e-10, atol=1e-14
            )
