import jax
import numpy as np
from ase.build import bulk
from jax_md import space

from apax.config.model_config import GMNNConfig
from apax.data.preprocessing import compute_nl
from apax.utils.convert import atoms_to_inputs


def test_jaxmd_fractional_path_matches_cartesian_path():
    """The training path (Cartesian positions + periodic shifts) and the jax-md
    inference path (fractional positions, jax-md displacement) agree, including
    property heads that use absolute positions."""
    atoms = bulk("Cu", cubic=True).repeat((3, 3, 3))
    atoms.set_cell(atoms.cell.array @ np.array([[1, 0.05, 0], [0, 1, 0], [0, 0.03, 1]]))
    atoms.rattle(0.1, seed=0)

    inputs = atoms_to_inputs([atoms.copy()])
    R, box, Z = inputs["positions"][0], inputs["box"][0], atoms.numbers
    r_max = 4.5
    idx, offsets = compute_nl(R, box, r_max)
    assert np.any(np.abs(offsets) > 0)  # periodic images contribute

    config = GMNNConfig(
        nn=[16, 16],
        basis={"name": "gaussian", "n_basis": 5, "r_max": r_max},
        descriptor_dtype="fp64",
        readout_dtype="fp64",
        property_heads=[{"name": "dipole", "mode": "l1", "aggregation": "sum"}],
    ).model_dump()
    builder = GMNNConfig.get_builder(None)(config)

    cart_model = builder.build_energy_model(init_box=box)
    params = cart_model.init(jax.random.PRNGKey(0), R, Z, idx, box, offsets)
    e_cart, props_cart = cart_model.apply(params, R, Z, idx, box, offsets)

    displacement, _ = space.periodic_general(box, fractional_coordinates=True)
    md_model = builder.build_energy_model(init_box=box, inference_disp_fn=displacement)
    R_frac = space.transform(np.linalg.inv(box), R)
    e_md, props_md = md_model.apply(params, R_frac, Z, idx, box, np.zeros_like(offsets))

    np.testing.assert_allclose(e_md, e_cart, rtol=1e-10)
    np.testing.assert_allclose(props_md["dipole"], props_cart["dipole"], rtol=1e-8)
