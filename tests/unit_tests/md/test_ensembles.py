import jax
import jax.numpy as jnp
import pytest
from ase import units
from jax_md import space

from apax.config.md_config import NPTOptions, NVEOptions, NVTOptions
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


def test_npt_rejects_constraints():
    ensemble = NPTOptions(name="npt", dt=0.5, temperature_schedule=TEMPERATURE)
    with pytest.raises(NotImplementedError):
        get_ensemble(ensemble, make_sim_fns(), n_constrained_dof=3)
