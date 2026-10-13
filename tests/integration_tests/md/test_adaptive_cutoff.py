"""End-to-end adaptive cutoff: training, ASE calculator, and jax-md MD."""

import pathlib

import jax
import numpy as np
import pytest
import yaml
import znh5md
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write

from apax.config import Config, MDConfig
from apax.md.ase_calc import ASECalculator
from apax.md.io import H5TrajHandler
from apax.md.simulate import md_setup, run_sim
from apax.train.run import run

TEST_PATH = pathlib.Path(__file__).parent.resolve()

R_MAX = 5.0
NUM_NEIGHBORS = 12


def _cu(scale=1.0, rattle=0.05, seed=0):
    atoms = bulk("Cu", cubic=True).repeat((3, 3, 3))
    atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    atoms.rattle(rattle, seed=seed)
    atoms.calc = EMT()
    atoms.get_potential_energy()
    atoms.get_forces()
    return atoms


def _dimer(d):
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, d]])
    atoms.calc = EMT()
    atoms.get_potential_energy()
    atoms.get_forces()
    return atoms


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("adaptive")
    # dense periodic cells and gas-phase dimers in one dataset
    data = [_cu(scale=s, seed=i) for i, s in enumerate([0.95, 1.0, 1.05, 1.0, 0.97])]
    data = data + [_dimer(d) for d in [2.0, 3.0, 4.5]]
    data_path = path / "ds.extxyz"
    write(data_path, data)

    config = {
        "n_epochs": 2,
        "data": {
            "directory": path.as_posix(),
            "experiment": "model",
            "data_path": data_path.as_posix(),
            "n_train": 6,
            "n_valid": 2,
            "batch_size": 2,
            "valid_batch_size": 1,
        },
        "model": {
            "name": "gmnn",
            "nn": [16, 16],
            "n_radial": 3,
            "basis": {"name": "gaussian", "n_basis": 5, "r_max": R_MAX},
            "descriptor_dtype": "fp64",
            "readout_dtype": "fp64",
            "calc_stress": True,
            "adaptive_cutoff": {"num_neighbors": NUM_NEIGHBORS},
        },
        "loss": [{"name": "energy"}, {"name": "forces"}],
    }
    run(config)
    model_config = Config.model_validate(config)
    return model_config, path


def test_calculator_density_and_shadow_forces(trained_model):
    model_config, _ = trained_model
    calc = ASECalculator(model_config.data.model_version_path)

    atoms = _cu(scale=1.0)
    atoms.calc = calc
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    assert np.isfinite(E) and F.shape == (len(atoms), 3)
    k_dense = calc.k_sel

    # Forces include the shadow term: compare to finite differences.
    eps = 1e-4
    for a, c in [(0, 0), (5, 1), (17, 2)]:
        disp = atoms.copy()
        disp.calc = calc
        disp.positions[a, c] += eps
        e_plus = disp.get_potential_energy()
        disp.positions[a, c] -= 2 * eps
        e_minus = disp.get_potential_energy()
        np.testing.assert_allclose(-(e_plus - e_minus) / (2 * eps), F[a, c], atol=1e-5)

    # Bounded environments: an expanded crystal selects the same number of
    # neighbors (at larger distances), so k_sel stays put.
    sparse = _cu(scale=1.15)
    sparse.calc = ASECalculator(model_config.data.model_version_path)
    sparse.get_potential_energy()
    assert abs(sparse.calc.k_sel - k_dense) <= 2


def test_dimer_reaches_beyond_dense_cutoff(trained_model):
    """A gas-phase dimer interacts out to almost r_max (the per-atom cutoff
    extends to find neighbors), and not at all beyond it."""
    model_config, _ = trained_model
    calc = ASECalculator(model_config.data.model_version_path)

    energies = []
    for d in [3.5, 4.0, 4.5, 5.5]:
        dimer = _dimer(d)
        dimer.calc = calc
        energies.append(dimer.get_potential_energy())
        if d < 4.6:
            assert np.abs(dimer.get_forces()).max() > 1e-6
    # beyond r_max the two atoms no longer interact
    np.testing.assert_allclose(dimer.get_forces(), 0.0, atol=1e-12)
    assert len(set(np.round(energies[:3], 10))) == 3


def test_calculator_overflow_resizes(trained_model):
    model_config, _ = trained_model
    calc = ASECalculator(model_config.data.model_version_path)
    atoms = _cu(scale=1.0, rattle=0.0)
    atoms.calc = calc
    atoms.get_potential_energy()
    k_initial = calc.k_sel

    # Locally compress: atoms around atom 0 pull together, so atom 0's neighbors
    # see a denser environment than the capacity was sized for.
    atoms = atoms.copy()
    center = atoms.positions[0].copy()
    d = atoms.positions - center
    r = np.linalg.norm(d, axis=1)
    shell = (r > 0) & (r < 4.5)
    atoms.positions[shell] = center + 0.6 * d[shell]
    atoms.calc = calc
    E = atoms.get_potential_energy()
    assert calc.k_sel > k_initial

    fresh = atoms.copy()
    fresh.calc = ASECalculator(model_config.data.model_version_path)
    np.testing.assert_allclose(E, fresh.get_potential_energy(), rtol=1e-10)


def test_batch_eval(trained_model):
    model_config, _ = trained_model
    calc = ASECalculator(model_config.data.model_version_path)
    frames = [_cu(scale=s, seed=i) for i, s in enumerate([0.95, 1.1])]
    evaluated = calc.batch_eval(frames, batch_size=2, silent=True)
    for frame, ev in zip(frames, evaluated):
        frame.calc = ASECalculator(model_config.data.model_version_path)
        np.testing.assert_allclose(
            ev.get_potential_energy(), frame.get_potential_energy(), rtol=1e-8
        )


def test_md_adaptive_overflow_rebuild(trained_model):
    model_config, path = trained_model
    initial = path / "initial.extxyz"
    write(initial, _cu(scale=1.0, rattle=0.0))

    with open(TEST_PATH / "md_config.yaml", "r") as stream:
        md_dict = yaml.safe_load(stream)
    md_dict["sim_dir"] = (path / "md").as_posix()
    md_dict["initial_structure"] = initial.as_posix()
    md_dict["ensemble"]["temperature_schedule"]["T0"] = 3000
    md_dict["ensemble"]["dt"] = 2.0
    md_dict["duration"] = 40.0
    md_dict["n_inner"] = 5
    md_dict["sampling_rate"] = 5
    md_config = MDConfig.model_validate(md_dict)
    pathlib.Path(md_config.sim_dir).mkdir(parents=True, exist_ok=True)

    # Deliberately too small: the perfect lattice selects NUM_NEIGHBORS, thermal
    # motion then needs more and the chunk has to be redone with a larger k_sel.
    system, sim_fns = md_setup(model_config, md_config, k_sel=NUM_NEIGHBORS)
    resized = []

    def resize_sim_fns(max_count):
        resized.append(max_count)
        return md_setup(model_config, md_config, k_sel=max_count + 2)[1]

    traj_path = pathlib.Path(md_config.sim_dir) / md_config.traj_name
    traj_handler = H5TrajHandler(
        system,
        md_config.buffer_size,
        traj_path,
        md_config.ensemble.dt,
        properties=md_config.properties,
        h5md_options=md_config.h5md_options.model_dump(),
    )
    n_steps = int(np.ceil(md_config.duration / md_config.ensemble.dt))
    run_sim(
        system,
        sim_fns,
        md_config.ensemble,
        sim_dir=pathlib.Path(md_config.sim_dir),
        n_steps=n_steps,
        n_inner=md_config.n_inner,
        extra_capacity=0,
        rng_key=jax.random.PRNGKey(0),
        traj_handler=traj_handler,
        sampling_rate=md_config.sampling_rate,
        restart=False,
        disable_pbar=True,
        resize_sim_fns=resize_sim_fns,
    )
    assert resized, "expected at least one adaptive overflow"
    assert all(m > NUM_NEIGHBORS for m in resized)

    traj = znh5md.IO(traj_path.as_posix())[:]
    assert len(traj) == n_steps // md_config.sampling_rate
    assert np.all(np.isfinite(traj[-1].positions))
