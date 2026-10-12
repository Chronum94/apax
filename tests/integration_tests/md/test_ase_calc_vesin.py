import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
import yaml
from ase import Atoms
from ase.build import bulk, molecule

from apax.config import Config
from apax.md.ase_calc import ASECalculator, build_energy_neighbor_fns, get_step_fn
from apax.utils.space import perpendicular_widths, transform

TEST_PATH = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def model_dir(get_tmp_path):
    with open(TEST_PATH / "config.yaml", "r") as stream:
        config_dict = yaml.safe_load(stream)
    config_dict["data"]["directory"] = get_tmp_path.as_posix()
    config_dict["model"]["calc_stress"] = True
    config = Config.model_validate(config_dict)
    config.data.model_version_path.mkdir(parents=True, exist_ok=True)
    config.dump_config(config.data.model_version_path)

    builder = config.model.get_builder()(config.model.model_dump())
    model = builder.build_energy_derivative_model()
    idx = jnp.array([[0, 1], [1, 0]])
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.array([[0.0, 0, 0], [1.0, 0, 0]]),
        jnp.array([1, 8]),
        idx,
        jnp.zeros((3, 3)),
        offsets=jnp.zeros((2, 3)),
    )
    options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1)
    with ocp.CheckpointManager(config.data.best_model_path, options=options) as mngr:
        mngr.save(
            0, args=ocp.args.StandardSave({"model": {"params": params}, "epoch": 0})
        )
    return config.data.model_version_path


def systems():
    gas = molecule("CH3CH2OH")
    # triclinic and not upper triangular, but wide enough for jax_md's
    # minimum image convention at the default 6 A cutoff
    big = bulk("Cu", "fcc", a=3.6).repeat((7, 7, 7))
    big.cell[2] += [0.6, 0.3, 0.0]
    big.numbers[::2] = 8
    small = bulk("Cu", "fcc", a=2.2)  # cutoff > half the cell: needs several images
    small.numbers[:] = 8
    for atoms in (big, small):
        atoms.rattle(0.05, seed=2)
    return {"gas": gas, "big_triclinic": big, "small_cell": small}


def evaluate(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()


@pytest.mark.parametrize("name", ["gas", "big_triclinic", "small_cell"])
def test_calculator_matches_batch_eval(model_dir, name):
    atoms = systems()[name]
    calc = ASECalculator(model_dir)
    energy, forces, _ = evaluate(calc, atoms)

    (ref,) = calc.batch_eval([atoms], batch_size=1, silent=True)
    np.testing.assert_allclose(energy, ref.get_potential_energy(), rtol=1e-6)
    np.testing.assert_allclose(forces, ref.get_forces(), atol=1e-6)


@pytest.mark.parametrize("name", ["gas", "big_triclinic"])
def test_calculator_matches_jax_md_neighbors(model_dir, name):
    atoms = systems()[name]
    calc = ASECalculator(model_dir)
    energy, forces, _ = evaluate(calc, atoms)

    periodic = np.any(atoms.cell.array > 1e-6)
    if periodic:
        half_width = perpendicular_widths(atoms.cell.array.T).min() / 2
        assert calc.r_max < half_width, "jax_md reference needs the minimum image"
    model, neighbor_fn = build_energy_neighbor_fns(
        atoms, calc.model_config, calc.params, 0.5, True, calc_stress=True
    )
    step = get_step_fn(model, jnp.asarray(atoms.numbers), periodic, True)
    positions = jnp.asarray(atoms.positions)
    box = jnp.asarray(atoms.cell.array)
    if periodic:
        frac = transform(jnp.linalg.inv(box.T), positions)
        neighbors = neighbor_fn.allocate(frac, box=box.T)
    else:
        neighbors = neighbor_fn.allocate(positions)
    ref, _ = step(positions, neighbors, box)

    np.testing.assert_allclose(energy, ref["energy"], rtol=1e-6)
    np.testing.assert_allclose(forces, ref["forces"], atol=1e-6)
    if periodic:
        # the calculator divides by the volume and transposes (ProcessStress)
        ref_stress = np.asarray(ref["stress"]).T / np.linalg.det(np.asarray(box))
        np.testing.assert_allclose(calc.results["stress"], ref_stress, atol=1e-10)


def test_calculator_follows_structure_changes(model_dir):
    calc = ASECalculator(model_dir)
    big = systems()["big_triclinic"]
    atoms = big.copy()
    atoms.calc = calc
    atoms.get_potential_energy()

    # moved atoms, deformed cell, then gas phase and back: each result must
    # match a fresh calculator
    moved = big.copy()
    moved.positions += np.random.default_rng(0).normal(
        scale=0.3, size=moved.positions.shape
    )
    deformed = moved.copy()
    deformed.set_cell(moved.cell.array * 0.97, scale_atoms=True)
    gas = systems()["gas"]

    for new in (moved, deformed, gas, big):
        atoms.set_cell(new.cell)
        atoms.set_pbc(new.pbc)
        if len(new) != len(atoms) or (new.numbers != atoms.numbers).any():
            atoms = new.copy()
            atoms.calc = calc
        atoms.positions = new.positions
        reused = atoms.get_potential_energy(), atoms.get_forces()
        fresh = evaluate(ASECalculator(model_dir), new)[:2]
        np.testing.assert_allclose(reused[0], fresh[0], rtol=1e-6)
        np.testing.assert_allclose(reused[1], fresh[1], atol=1e-6)


def test_gas_phase_has_no_cell(model_dir):
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    energy, forces, _ = evaluate(ASECalculator(model_dir), atoms)
    assert np.isfinite(energy) and np.all(np.isfinite(forces))
