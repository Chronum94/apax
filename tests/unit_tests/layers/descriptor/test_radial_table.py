import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk

from apax.config.model_config import GMNNConfig
from apax.data.preprocessing import compute_nl
from apax.layers.descriptor.radial_table import tabulate_radial
from apax.utils.convert import atoms_to_inputs

R_MAX = 5.0
ELEMENTS = (3, 9)  # Li, F


def _config(radial_function, radial_transform="identity"):
    return GMNNConfig(
        basis={"name": "bessel", "n_basis": 8, "r_max": R_MAX},
        nn=[16, 16],
        radial_function=radial_function,
        radial_transform=radial_transform,
        descriptor_dtype="fp64",
        readout_dtype="fp64",
    )


def _exact_radial(radial_function, radial_transform="identity"):
    cfg = _config(radial_function, radial_transform)
    radial = cfg.get_builder()(cfg.model_dump()).build_radial_function()
    d = jnp.linspace(1.0, 4.0, 3)
    z = jnp.full(d.shape, ELEMENTS[0])
    params = radial.init(jax.random.PRNGKey(1), d, z, z)["params"]
    return radial, params


@pytest.mark.parametrize("radial_transform", ["identity", "covalent"])
@pytest.mark.parametrize("radial_function", ["dense", "factorized"])
def test_tabulated_radial_matches_exact(radial_function, radial_transform):
    with jax.enable_x64(True):
        radial, params = _exact_radial(radial_function, radial_transform)
        table = tabulate_radial(radial, params, ELEMENTS, dtype=jnp.float64)
        # includes distances beyond the cutoff
        d = jnp.asarray(np.random.default_rng(0).uniform(0.5, R_MAX + 0.5, 4000))
        for zc in ELEMENTS:
            for zn in ELEMENTS:
                zc_, zn_ = jnp.full(d.shape, zc), jnp.full(d.shape, zn)

                def exact(x):
                    return radial.apply({"params": params}, x, zn_, zc_)

                def approx(x):
                    return table.apply({}, x, zn_, zc_)

                (r_ex, dr_ex), (r_ap, dr_ap) = (
                    jax.jvp(f, (d,), (jnp.ones_like(d),)) for f in (exact, approx)
                )
                assert jnp.abs(r_ap - r_ex).max() < 1e-11 * jnp.abs(r_ex).max()
                assert jnp.abs(dr_ap - dr_ex).max() < 1e-10 * jnp.abs(dr_ex).max()


@pytest.mark.parametrize("radial_transform", ["identity", "covalent"])
def test_tabulated_model_energy_forces(radial_transform):
    atoms = bulk("LiF", "rocksalt", a=4.03).repeat(2)
    atoms.rattle(0.1, seed=0)
    with jax.enable_x64(True):
        cfg = _config("factorized", radial_transform)
        inp = atoms_to_inputs([atoms])
        R, Z = np.asarray(inp["positions"][0]), np.asarray(inp["numbers"][0])
        box = np.asarray(inp["box"][0])
        idx, offsets = compute_nl(R, box, R_MAX)
        builder = cfg.get_builder()(cfg.model_dump())
        model = builder.build_energy_derivative_model(init_box=box)
        params = model.init(jax.random.PRNGKey(0), R, Z, idx, box, offsets)
        exact = model.apply(params, R, Z, idx, box, offsets)

        radial_params = params["params"]["energy_model"]["representation"]["radial_fn"]
        table = tabulate_radial(
            builder.build_radial_function(), radial_params, ELEMENTS, dtype=jnp.float64
        )
        tab_builder = cfg.get_builder()(cfg.model_dump(), radial_fn=table)
        tab_model = tab_builder.build_energy_derivative_model(init_box=box)
        approx = tab_model.apply(params, R, Z, idx, box, offsets)

        np.testing.assert_allclose(approx["energy"], exact["energy"], rtol=1e-9)
        f_scale = np.abs(exact["forces"]).max()
        np.testing.assert_allclose(approx["forces"], exact["forces"], atol=1e-7 * f_scale)


@pytest.mark.parametrize("radial_transform", ["identity", "covalent"])
def test_table_dtype_follows_target_not_sampler(radial_transform):
    """Sampling happens in float64; the table, its evaluation basis and its output must
    be in the target dtype."""
    with jax.enable_x64(True):
        radial, params = _exact_radial("factorized", radial_transform)
        table = tabulate_radial(radial, params, ELEMENTS, dtype=jnp.float32)
        d = jnp.linspace(1.0, 4.0, 5, dtype=jnp.float32)
        z = jnp.full(d.shape, ELEMENTS[0])
        assert table.apply({}, d, z, z).dtype == jnp.float32
