"""apax md (jax-md driver) invariants: what lands in md.h5 must match the model, the clock and a fresh run."""

from pathlib import Path
from unittest.mock import patch

import h5py
import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest
import znh5md
from ase.io import write

import apax.md.simulate as simulate
from apax.config import Config
from apax.md import run_md
from apax.md.ase_calc import ASECalculator
from tests.invariants.conftest import build, model_config, random_bulk

R_MAX = 3.5  # cubic Cu rep=2 (7.2 Å) is thick, rep=1 (3.6 Å) is thin
DT = 0.5


# ---------------------------------------------------------------- helpers (conftest candidates)
def write_model_dir(tmp: Path, atoms, **cfg_overrides) -> dict:
    """Random-weight fp64 model saved as a real model directory. Returns the Config dict."""
    cfg = model_config(basis={"name": "bessel", "n_basis": 8, "r_max": R_MAX}, **cfg_overrides)
    cfg_dict = {
        "n_epochs": 1,
        "data": {"directory": str(tmp), "experiment": "model", "data_path": "unused"},
        "model": cfg.model_dump(),
        "loss": [{"name": "energy"}],
    }
    config = Config.model_validate(cfg_dict)
    with jax.enable_x64(True):
        _, params = build(cfg, atoms)
    config.data.model_version_path.mkdir(parents=True, exist_ok=True)
    config.dump_config(config.data.model_version_path)
    with ocp.CheckpointManager(config.data.best_model_path.resolve()) as mngr:
        mngr.save(0, args=ocp.args.StandardSave({"model": {"params": params}, "epoch": 0}))
    return cfg_dict


def md(tmp: Path, model: dict, atoms, name="sim", **kw):
    """Run apax md on `atoms` in tmp/name; returns the sim dir."""
    write(tmp / f"{name}.extxyz", atoms)
    cfg = dict(
        ensemble={"name": "nvt", "dt": DT}, duration=10.0, n_inner=4, sampling_rate=3, buffer_size=5,
        initial_structure=str(tmp / f"{name}.extxyz"), sim_dir=str(tmp / name), disable_pbar=True,
        properties=["energy", "forces", "stress"],
    )
    cfg.update(kw)
    run_md(model, cfg)
    return tmp / name


def frames(sim):
    return znh5md.IO(sim / "md.h5")[:]


def h5(sim, key):
    with h5py.File(sim / "md.h5") as f:
        return f[key][()]


def ase_results(model, atoms_list, props=("energy", "forces")):
    calc = ASECalculator(Path(model["data"]["directory"]) / "model")
    out = {p: [] for p in props}
    for a in atoms_list:
        a = a.copy()
        a.calc = calc
        for p in props:
            out[p].append(a.get_stress(voigt=False) if p == "stress" else a.calc.get_property(p, a))
    return {p: np.array(v) for p, v in out.items()}


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("md")


@pytest.fixture(scope="module")
def cu():
    return random_bulk("cubic", rep=2, species=(29,), rattle=0.05)


@pytest.fixture(scope="module")
def model(tmp, cu):
    return write_model_dir(tmp / "single", cu)


@pytest.fixture(scope="module")
def run(tmp, model, cu):
    """duration 10 fs / dt 0.5 = 20 steps = 5 n_inner blocks; sampling_rate 3 → steps 0,3,..,18 → 7 frames."""
    with jax.enable_x64(True):
        return md(tmp, model, cu)


# ---------------------------------------------------------------- md.h5 contents
def test_frame_count(run):
    n_steps = int(np.ceil(np.ceil(10.0 / DT) / 4) * 4)
    assert len(frames(run)) == int(np.ceil(n_steps / 3)) == 7


def test_stored_energy_forces_match_ase_calculator(run, model):
    traj = frames(run)
    ref = ase_results(model, traj)
    np.testing.assert_allclose([a.get_potential_energy() for a in traj], ref["energy"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose([a.get_forces() for a in traj], ref["forces"], rtol=1e-8, atol=1e-8)


@pytest.mark.xfail(strict=True, reason="md.h5 time/step arrays count frames, not MD steps (sampling_rate ignored)")
def test_time_and_step_arrays(run):
    # frame k holds the state after integration step k*sampling_rate (0-based), i.e. step k*sr+1
    step = h5(run, "particles/atoms/position/step")
    time = h5(run, "particles/atoms/position/time")
    expected = np.arange(7) * 3 + 1
    np.testing.assert_array_equal(step, expected)
    np.testing.assert_allclose(time, expected * DT)


@pytest.mark.xfail(strict=True, reason="md.h5 time/step arrays are shorter than the data once the buffer flushes twice")
def test_time_array_length_matches_frames(run):
    assert len(h5(run, "particles/atoms/position/time")) == len(h5(run, "particles/atoms/position/value"))


@pytest.mark.xfail(strict=True, reason="apax md stores stress × volume (eV), not stress (eV/Å^3) like ASECalculator")
def test_stored_stress_matches_ase_calculator(run, model):
    traj = frames(run)
    ref = ase_results(model, traj, props=("stress",))["stress"]
    np.testing.assert_allclose(h5(run, "observables/atoms/stress/value"), ref, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------- restarts
@pytest.fixture(scope="module")
def restarted(tmp, model, cu):
    """(fresh 2T run, T run extended to 2T)."""
    kw = dict(n_inner=2, sampling_rate=2, buffer_size=3, checkpoint_interval=1)
    with jax.enable_x64(True):
        fresh = md(tmp, model, cu, name="fresh", duration=8.0, **kw)
        md(tmp, model, cu, name="ext", duration=4.0, **kw)
        ext = md(tmp, model, cu, name="ext", duration=8.0, **kw)
    return fresh, ext


def test_restart_extends_trajectory_identically(restarted):
    fresh, ext = map(frames, restarted)
    assert len(ext) == len(fresh) == 8
    np.testing.assert_allclose([a.positions for a in ext], [a.positions for a in fresh], rtol=0, atol=1e-10)
    np.testing.assert_allclose([a.get_momenta() for a in ext], [a.get_momenta() for a in fresh], rtol=0, atol=1e-10)


@pytest.mark.xfail(strict=True, reason="time/step arrays restart from 0 after an MD restart")
def test_restart_time_arrays_identical(restarted):
    fresh, ext = restarted
    for key in ("particles/atoms/position/step", "particles/atoms/position/time"):
        np.testing.assert_array_equal(h5(ext, key), h5(fresh, key))


@pytest.mark.xfail(strict=True, reason="restart after a crash truncates md.h5 to step*n_inner frames, ignoring sampling_rate")
def test_restart_after_crash_matches_fresh_run(tmp, model, cu, restarted):
    kw = dict(n_inner=2, sampling_rate=2, buffer_size=1, checkpoint_interval=2, duration=8.0)
    real_check = simulate.check_for_nans

    def crash_at_block_3(state, step):  # ckpt exists after block 2; frames of blocks 0..3 are on disk
        if step == 3:
            raise KeyboardInterrupt
        real_check(state, step)

    with jax.enable_x64(True):
        with patch.object(simulate, "check_for_nans", crash_at_block_3), pytest.raises(KeyboardInterrupt):
            md(tmp, model, cu, name="crash", **kw)
        crashed = md(tmp, model, cu, name="crash", **kw)
    fresh = frames(restarted[0])
    resumed = frames(crashed)
    assert len(resumed) == len(fresh)
    np.testing.assert_allclose([a.positions for a in resumed], [a.positions for a in fresh], rtol=0, atol=1e-10)


def test_run_md_twice_in_one_process(tmp, model, cu, run):
    with jax.enable_x64(True):
        again = md(tmp, model, cu, name="again")
    np.testing.assert_array_equal(h5(again, "particles/atoms/position/value"), h5(run, "particles/atoms/position/value"))


# ---------------------------------------------------------------- integrators
@pytest.mark.xfail(strict=True, raises=TypeError, reason="ensemble name 'nve' crashes: simulate.nve called with (kT, dt)")
def test_nve_conserves_energy(tmp, model, cu):
    with jax.enable_x64(True):
        sim = md(tmp, model, cu, name="nve", ensemble={"name": "nve", "dt": DT}, duration=50.0,
                 n_inner=10, sampling_rate=10)
    traj = frames(sim)
    e = np.array([a.get_potential_energy() + a.get_kinetic_energy() for a in traj])
    assert np.ptp(e) < 1e-3 * len(cu)


@pytest.mark.xfail(strict=True, reason="NVT thermostat_chain options (tau, chain_length, ...) are ignored")
def test_nvt_thermostat_tau_is_used(tmp, model, cu):
    with jax.enable_x64(True):
        a, b = (
            md(tmp, model, cu, name=f"tau{tau}", ensemble={"name": "nvt", "dt": DT, "thermostat_chain": {"tau": tau}})
            for tau in (10, 10_000)
        )
    assert not np.allclose(h5(a, "particles/atoms/velocity/value"), h5(b, "particles/atoms/velocity/value"))


# ---------------------------------------------------------------- thin box
@pytest.mark.xfail(strict=True, reason="apax md misses periodic images when the cell is thinner than 2 r_max")
def test_thin_box_matches_ase_calculator(tmp):
    thin = random_bulk("cubic", rep=1, species=(29,), rattle=0.1)
    model = write_model_dir(tmp / "thin_model", thin)
    with jax.enable_x64(True):
        traj = frames(md(tmp, model, thin, name="thin", duration=2.0, n_inner=1, sampling_rate=1))
    ref = ase_results(model, traj)
    np.testing.assert_allclose([a.get_potential_energy() for a in traj], ref["energy"], rtol=1e-8)


# ---------------------------------------------------------------- dynamics checks
def test_forces_uncertainty_check_stops_run(tmp, cu):
    model = write_model_dir(tmp / "shallow", cu, ensemble={"kind": "shallow", "n_members": 4, "force_variance": True})
    with jax.enable_x64(True):
        sim = md(tmp, model, cu, name="unc", n_inner=4, sampling_rate=1,
                 dynamics_checks=[{"name": "forces_uncertainty", "threshold": 1e-12}])
    # the first sampled frame fails; only that n_inner block (4 frames) may be on disk, not the 20 of a full run
    assert 1 <= len(frames(sim)) <= 4
    assert "dynamics checks failed" in (sim / "md.log").read_text()
