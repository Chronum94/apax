"""Derivative invariants for any weights: forces = -dE/dR, Hessian = -dF/dR, momentum/torque conservation,
shallow-ensemble bookkeeping, NVE energy conservation.

Stress vs FD for periodic cells lives in tests/unit_tests/layers/test_periodic_cells.py; a cluster has no
volume, so cluster stress is not tested.
"""

import jax
import numpy as np
import pytest
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from hypothesis import given
from hypothesis import strategies as st

from tests.invariants.conftest import (
    ase_calculator,
    ase_eval,
    build,
    fd_forces,
    model_config,
    offset_path,
    random_bulk,
    random_cluster,
)

SYSTEMS = {
    "cluster": lambda seed: random_cluster(8, seed=seed),
    "thin_triclinic": lambda seed: random_bulk("triclinic", rep=1, seed=seed),
    "skewed": lambda seed: random_bulk("skewed", rep=1, seed=seed),
}


def energy_fn(builder, params, key="energy"):
    return lambda a: offset_path(builder, params, a)[key]


# ---------------------------------------------------------------- forces
@pytest.mark.parametrize("system", SYSTEMS)
@given(seed=st.integers(0, 2**16))
def test_training_path_forces_match_fd(system, seed):
    atoms = SYSTEMS[system](seed)
    builder, params = build(model_config(calc_stress=False), atoms, seed=seed)
    forces = offset_path(builder, params, atoms)["forces"]
    np.testing.assert_allclose(forces, fd_forces(energy_fn(builder, params), atoms), atol=1e-7)
    np.testing.assert_allclose(forces.sum(0), 0, atol=1e-10)  # Newton's third law, every system
    if not atoms.pbc.any():
        torque = np.cross(atoms.positions - atoms.positions.mean(0), forces).sum(0)
        np.testing.assert_allclose(torque, 0, atol=1e-10)


@pytest.mark.parametrize("system", SYSTEMS)
def test_ase_forces_match_fd(system):
    atoms = SYSTEMS[system](1)
    cfg = model_config(calc_stress=False)
    _, params = build(cfg, atoms)
    calc = ase_calculator(cfg, params)
    forces = ase_eval(calc, atoms)["forces"]
    np.testing.assert_allclose(forces, fd_forces(lambda a: ase_eval(calc, a)["energy"], atoms), atol=1e-7)
    np.testing.assert_allclose(forces.sum(0), 0, atol=1e-10)


# ---------------------------------------------------------------- hessian
@pytest.mark.parametrize("system", ["cluster", "thin_triclinic"])
def test_ase_hessian(system):
    atoms = SYSTEMS[system](2)
    cfg = model_config(calc_stress=False)
    _, params = build(cfg, atoms)
    calc = ase_calculator(cfg, params, calc_hessian=True)
    a = atoms.copy()
    a.calc = calc
    H = calc.get_hessian(a)

    # H = -dF/dR by central differences on the calculator's own forces
    h, n = 1e-5, len(atoms)
    fd = np.zeros((3 * n, 3 * n))
    for i in range(3 * n):
        f = []
        for s in (1, -1):
            b = atoms.copy()
            b.positions[i // 3, i % 3] += s * h
            f.append(ase_eval(calc, b)["forces"].ravel())
        fd[i] = -(f[0] - f[1]) / (2 * h)
    np.testing.assert_allclose(H, fd, atol=1e-6)
    np.testing.assert_allclose(H, H.T, atol=1e-10)

    # acoustic sum rule: rigid translation leaves forces unchanged, sum_j H[i, j] = 0 per direction
    np.testing.assert_allclose(H.reshape(n, 3, n, 3).sum(2), 0, atol=1e-9)
    # ponytail: rotational null space only holds at a stationary point (H·(ω×r) = -ω×F), checked here
    # as the identity itself instead of relaxing to a minimum
    if not atoms.pbc.any():
        r = atoms.positions - atoms.positions.mean(0)
        F = ase_eval(calc, atoms)["forces"]
        for w in np.eye(3):
            np.testing.assert_allclose(H @ np.cross(w, r).ravel(), -np.cross(w, F).ravel(), atol=1e-9)


# ---------------------------------------------------------------- shallow ensemble
N_MEMBERS = 4


def shallow(atoms, identical=False):
    cfg = model_config(calc_stress=False, ensemble={"kind": "shallow", "n_members": N_MEMBERS})
    builder, params = build(cfg, atoms)
    if identical:
        params = jax.tree_util.tree_map_with_path(
            lambda p, x: x[:, :1].repeat(N_MEMBERS, 1) if "dense_2" in jax.tree_util.keystr(p) else x, params
        )
    return cfg, builder, params


@pytest.mark.parametrize("system", ["cluster", "thin_triclinic"])
def test_shallow_ensemble_derivatives(system):
    atoms = SYSTEMS[system](3)
    _, builder, params = shallow(atoms)
    out = offset_path(builder, params, atoms)
    E, F = out["energy_ensemble"], out["forces_ensemble"]
    assert E.shape == (N_MEMBERS,) and F.shape == (len(atoms), 3, N_MEMBERS)
    assert np.ptp(E) > 1e-3  # members actually differ

    np.testing.assert_allclose(out["energy"], E.mean(), rtol=1e-12)
    np.testing.assert_allclose(out["forces"], F.mean(-1), atol=1e-12)
    np.testing.assert_allclose(out["energy_uncertainty"], E.std(ddof=1), rtol=1e-10)
    np.testing.assert_allclose(out["forces_uncertainty"], F.std(-1, ddof=1), atol=1e-10)
    np.testing.assert_allclose(out["forces"], fd_forces(energy_fn(builder, params), atoms), atol=1e-7)
    member0 = lambda a: offset_path(builder, params, a)["energy_ensemble"][0]  # noqa: E731
    np.testing.assert_allclose(F[..., 0], fd_forces(member0, atoms), atol=1e-7)
    np.testing.assert_allclose(F.sum(0), 0, atol=1e-10)


def test_shallow_ensemble_identical_members_have_zero_uncertainty():
    atoms = SYSTEMS["cluster"](4)
    _, builder, params = shallow(atoms, identical=True)
    out = offset_path(builder, params, atoms)
    assert abs(out["energy"]) > 1e-3
    np.testing.assert_allclose(out["energy_uncertainty"], 0, atol=1e-10)
    np.testing.assert_allclose(out["forces_uncertainty"], 0, atol=1e-10)


def test_shallow_ensemble_ase_matches_training_path():
    atoms = SYSTEMS["thin_triclinic"](5)
    cfg, builder, params = shallow(atoms)
    ref = offset_path(builder, params, atoms)
    a = atoms.copy()
    a.calc = ase_calculator(cfg, params)
    a.get_potential_energy()
    for k in ("energy", "forces", "energy_uncertainty", "forces_uncertainty", "energy_ensemble", "forces_ensemble"):
        np.testing.assert_allclose(a.calc.results[k], ref[k], atol=1e-10, err_msg=k)


# ---------------------------------------------------------------- NVE
def test_nve_energy_conservation():
    atoms = random_cluster(10, min_dist=1.5, seed=6)
    cfg = model_config(calc_stress=False)
    _, params = build(cfg, atoms)
    atoms.calc = ase_calculator(cfg, params)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))
    etot = lambda: atoms.get_potential_energy() + atoms.get_kinetic_energy()  # noqa: E731
    e0, dyn, energies = etot(), VelocityVerlet(atoms, timestep=0.1 * units.fs), []
    for _ in range(50):
        dyn.run(4)
        energies.append(etot())
    # Verlet error is O(dt^2) and bounded; a wrong force (e.g. sign or missing term) drifts linearly
    assert np.max(np.abs(np.array(energies) - e0)) < 1e-4 * max(1.0, abs(e0))
