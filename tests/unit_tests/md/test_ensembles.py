import jax
import jax.numpy as jnp
import pytest
from ase import units
from jax_md import space

from apax.config.md_config import (
    LangevinOptions,
    NPTOptions,
    NVEOptions,
    NVTOptions,
)
from apax.md.sim_utils import SimulationFunctions
from apax.md.simulate import get_ensemble

N_ATOMS = 4
TEMPERATURE = {"name": "constant", "T0": 300}


def harmonic_energy(R, **kwargs):
    return 0.5 * jnp.sum(R**2)


def make_sim_fns():
    _, shift_fn = space.free()
    return SimulationFunctions(harmonic_energy, None, shift_fn, None)


def init_state(ensemble, n_constrained_dof=0):
    init_fn, apply_fn, _, _ = get_ensemble(ensemble, make_sim_fns(), n_constrained_dof)
    R = jax.random.normal(jax.random.PRNGKey(0), (N_ATOMS, 3))
    state = init_fn(jax.random.PRNGKey(1), R, mass=jnp.ones(N_ATOMS))
    return state, apply_fn


def test_nve_runs():
    ensemble = NVEOptions(name="nve", dt=0.5, temperature_schedule=TEMPERATURE)
    state, apply_fn = init_state(ensemble)

    state = apply_fn(state)

    assert jnp.all(jnp.isfinite(state.position))
    assert jnp.any(state.momentum != 0)


def test_nvt_uses_thermostat_chain_options():
    chain = {"chain_length": 2, "chain_steps": 1, "sy_steps": 1, "tau": 20}
    ensemble = NVTOptions(
        name="nvt", dt=0.5, temperature_schedule=TEMPERATURE, thermostat_chain=chain
    )
    state, apply_fn = init_state(ensemble)

    assert state.chain.position.shape == (2,)
    assert jnp.isclose(state.chain.tau, 20 * 0.5 * units.fs)
    apply_fn(state)


def test_nvt_removes_constrained_dof():
    ensemble = NVTOptions(name="nvt", dt=0.5, temperature_schedule=TEMPERATURE)

    state, _ = init_state(ensemble)
    assert state.chain.degrees_of_freedom == 3 * N_ATOMS

    state, apply_fn = init_state(ensemble, n_constrained_dof=3)
    assert state.chain.degrees_of_freedom == 3 * N_ATOMS - 3
    state = apply_fn(state)
    assert state.chain.degrees_of_freedom == 3 * N_ATOMS - 3


def test_langevin_samples_target_temperature():
    ensemble = LangevinOptions(
        name="langevin", dt=1.0, temperature_schedule=TEMPERATURE, friction=0.05
    )
    state, apply_fn = init_state(ensemble)
    # harmonic wells with unit mass: period ~64 fs in ASE time units
    n_steps = 20_000

    def body(state, _):
        state = apply_fn(state)
        ke = 0.5 * jnp.sum(state.momentum**2 / state.mass)
        return state, ke

    _, ke = jax.lax.scan(body, state, None, length=n_steps)
    temperature = 2 * jnp.mean(ke[n_steps // 10 :]) / (3 * N_ATOMS) / units.kB
    assert abs(temperature - 300) < 30


def test_langevin_keeps_fixed_atoms_in_place():
    ensemble = LangevinOptions(name="langevin", dt=1.0, temperature_schedule=TEMPERATURE)

    def fix_first_atom(state):
        return state.set(
            momentum=state.momentum.at[0].set(0.0), force=state.force.at[0].set(0.0)
        )

    init_fn, apply_fn, _, _ = get_ensemble(ensemble, make_sim_fns(), 0, fix_first_atom)
    R = jax.random.normal(jax.random.PRNGKey(0), (N_ATOMS, 3))
    state = fix_first_atom(init_fn(jax.random.PRNGKey(1), R, mass=jnp.ones(N_ATOMS)))

    state = fix_first_atom(apply_fn(state))

    assert jnp.allclose(state.position[0], R[0])
    assert not jnp.allclose(state.position[1:], R[1:])


def test_npt_rejects_constraints():
    ensemble = NPTOptions(name="npt", dt=0.5, temperature_schedule=TEMPERATURE)
    with pytest.raises(NotImplementedError):
        get_ensemble(ensemble, make_sim_fns(), n_constrained_dof=3)
