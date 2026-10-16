import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk

from apax.config.model_config import GMNNConfig
from apax.data.preprocessing import compute_nl
from apax.utils.convert import atoms_to_inputs

R_MAX = 5.0
N_MEMBERS = 4


def _inputs(atoms):
    inp = atoms_to_inputs([atoms])
    R, box = np.asarray(inp["positions"][0]), np.asarray(inp["box"][0])
    idx, offsets = compute_nl(R, box, R_MAX)
    return R, np.asarray(inp["numbers"][0]), idx, box, offsets


def _builder(**ensemble):
    cfg = GMNNConfig(
        basis={"name": "bessel", "n_basis": 8, "r_max": R_MAX},
        nn=[16, 16],
        calc_stress=True,
        descriptor_dtype="fp64",
        readout_dtype="fp64",
        ensemble={"kind": "shallow", "n_members": N_MEMBERS, **ensemble},
    )
    return cfg.get_builder()(cfg.model_dump())


def _reference(builder, params, R, Z, idx, box, off):
    """Per-member forces and stress from an explicit Jacobian of the member energies."""
    energy_model = builder.build_energy_model(init_box=box)
    eparams = {"params": params["params"]["energy_model"]}

    def member_energies(R, F):
        return energy_model.apply(eparams, R, Z, idx, box, off, F)[0]

    dR, dF = jax.jacrev(member_energies, argnums=(0, 1))(R, jnp.eye(3))
    return member_energies(R, jnp.eye(3)), -dR, 0.5 * (dF + jnp.swapaxes(dF, 1, 2))


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_shallow_ensemble_matches_member_jacobian(chunk_size):
    atoms = bulk("NaCl", "rocksalt", a=5.64).repeat(2)
    atoms.rattle(0.05, seed=0)
    with jax.enable_x64(True):
        builder = _builder(chunk_size=chunk_size)
        R, Z, idx, box, off = _inputs(atoms)
        model = builder.build_energy_derivative_model(init_box=box)
        params = model.init(jax.random.PRNGKey(0), R, Z, idx, box, off)
        pred = model.apply(params, R, Z, idx, box, off)

        e_ens, f_ens, s_ens = _reference(builder, params, R, Z, idx, box, off)
        ddof = 1
        np.testing.assert_allclose(pred["energy_ensemble"], e_ens, rtol=1e-12)
        np.testing.assert_allclose(pred["energy"], e_ens.mean(), rtol=1e-12)
        np.testing.assert_allclose(
            pred["energy_uncertainty"], jnp.std(e_ens, ddof=ddof), rtol=1e-10
        )
        np.testing.assert_allclose(
            pred["forces_ensemble"], jnp.transpose(f_ens, (1, 2, 0)), atol=1e-12
        )
        np.testing.assert_allclose(pred["forces"], f_ens.mean(0), atol=1e-12)
        np.testing.assert_allclose(
            pred["forces_uncertainty"], jnp.std(f_ens, axis=0, ddof=ddof), atol=1e-10
        )
        np.testing.assert_allclose(pred["stress"], s_ens.mean(0), atol=1e-12)
        np.testing.assert_allclose(
            pred["stress_ensemble"], jnp.transpose(s_ens, (1, 2, 0)), atol=1e-12
        )
        np.testing.assert_allclose(
            pred["stress_uncertainty"], jnp.std(s_ens, axis=0, ddof=ddof), atol=1e-10
        )


def test_shallow_ensemble_mean_only():
    atoms = bulk("NaCl", "rocksalt", a=5.64).repeat(2)
    atoms.rattle(0.05, seed=1)
    with jax.enable_x64(True):
        builder = _builder()
        R, Z, idx, box, off = _inputs(atoms)
        model = builder.build_energy_derivative_model(init_box=box, force_variance=False)
        params = model.init(jax.random.PRNGKey(0), R, Z, idx, box, off)
        pred = model.apply(params, R, Z, idx, box, off)

        e_ens, f_ens, s_ens = _reference(builder, params, R, Z, idx, box, off)
        np.testing.assert_allclose(pred["energy"], e_ens.mean(), rtol=1e-12)
        np.testing.assert_allclose(pred["forces"], f_ens.mean(0), atol=1e-12)
        np.testing.assert_allclose(pred["stress"], s_ens.mean(0), atol=1e-12)
        assert "forces_uncertainty" not in pred
