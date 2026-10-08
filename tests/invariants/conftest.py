"""Shared helpers for invariant / reference tests: random-weight models on random structures, fp64.

These tests must hold for ANY weights and ANY configuration. Use them, extend them here if several files need it.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from apax.config.model_config import GMNNConfig
from apax.config.train_config import Config
from apax.data.preprocessing import compute_nl
from apax.md.ase_calc import ASECalculator
from apax.utils.convert import atoms_to_inputs

R_MAX = 5.0
ENERGY_SCALE = 30.0  # random init gives tiny energies; scale outputs so E, F, stress are O(1)


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


# ---------------------------------------------------------------- structures
def random_cluster(n=12, species=(1, 6, 8), min_dist=1.0, box=6.0, seed=0) -> Atoms:
    """Non-periodic point cloud with a minimum pair distance."""
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < n:
        p = rng.uniform(0, box, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in pos):
            pos.append(p)
    return Atoms(numbers=rng.choice(species, n), positions=pos)


def deformed_fcc(F, rep, a=3.6, element="Cu"):
    atoms = bulk(element, "fcc", a=a, cubic=True).repeat(rep)
    atoms.set_cell(atoms.cell.array @ np.asarray(F).T, scale_atoms=True)
    return atoms


CELL_SHAPES = {
    "cubic": np.eye(3),
    "tetragonal": np.diag([1.0, 1.0, 1.08]),
    "orthorhombic": np.diag([1.0, 1.1, 0.93]),
    "monoclinic": [[1, 0, 0.25], [0, 1, 0], [0, 0, 1]],
    "triclinic": [[1, 0.15, 0.2], [0.05, 1, 0.1], [0.12, 0.07, 1]],
    "skewed": [[1, 0.5, 0.4], [0, 1, 0.45], [0, 0, 1]],
}


def random_bulk(shape="triclinic", rep=2, species=(29, 79), seed=0, rattle=0.05) -> Atoms:
    """Periodic cell of a given shape, random species, rattled. rep=2 → thin (< 2 r_max), rep=3 → thick."""
    atoms = deformed_fcc(CELL_SHAPES[shape], rep)
    rng = np.random.default_rng(seed)
    atoms.numbers = rng.choice(species, len(atoms))
    atoms.rattle(rattle, seed=seed)
    return atoms


def min_height(cell):
    cell = np.asarray(cell)
    vol = abs(np.linalg.det(cell))
    a, b, c = cell
    return min(vol / np.linalg.norm(np.cross(x, y)) for x, y in ((b, c), (c, a), (a, b)))


# ---------------------------------------------------------------- models
def model_config(**overrides) -> GMNNConfig:
    kw = dict(
        basis={"name": "bessel", "n_basis": 8, "r_max": R_MAX},
        nn=[16, 16],
        calc_stress=True,
        descriptor_dtype="fp64",
        readout_dtype="fp64",
    )
    kw.update(overrides)
    return GMNNConfig(**kw)


def inputs(atoms):
    """Model inputs exactly as the training pipeline builds them (fractional R + box = cell.T for PBC)."""
    inp = atoms_to_inputs([atoms])
    R, box = np.asarray(inp["positions"][0]), np.asarray(inp["box"][0])
    idx, offsets = compute_nl(R, box, R_MAX)
    return R, np.asarray(inp["numbers"][0]), idx, box, offsets


def build(cfg, atoms, seed=0, scale=ENERGY_SCALE):
    """Returns (builder, params) for an EnergyDerivativeModel with random, rescaled weights."""
    builder = cfg.get_builder()(cfg.model_dump())
    inp = inputs(atoms)
    params = builder.build_energy_derivative_model(init_box=inp[3]).init(jax.random.PRNGKey(seed), *inp)
    params = jax.tree_util.tree_map_with_path(
        lambda p, x: x * scale if "scale_per_element" in jax.tree_util.keystr(p) else x, params
    )
    return builder, params


def offset_path(builder, params, atoms) -> dict:
    """Training-path prediction: energy, forces, stress (eV/Å^3, divided by volume for PBC)."""
    inp = inputs(atoms)
    out = builder.build_energy_derivative_model(init_box=inp[3]).apply(params, *inp)
    res = {k: np.asarray(v) for k, v in out.items()}
    if "stress" in res and atoms.pbc.all():
        res["stress"] = res["stress"] / atoms.get_volume()
    return res


def ase_calculator(cfg, params, **kwargs) -> ASECalculator:
    """Real ASECalculator around in-memory params (no model directory needed)."""
    config = MagicMock(spec=Config)
    config.model = cfg
    with patch("apax.md.ase_calc.restore_parameters", return_value=(config, params)):
        with patch("apax.md.ase_calc.check_for_ensemble", return_value=1):
            return ASECalculator(model_dir=Path("dummy"), **kwargs)


def ase_eval(calc, atoms, props=("energy", "forces")) -> dict:
    a = atoms.copy()
    a.calc = calc
    res = {"energy": a.get_potential_energy(), "forces": a.get_forces()}
    if "stress" in props:
        res["stress"] = a.get_stress(voigt=False)
    return res


# ---------------------------------------------------------------- finite differences
def fd_forces(energy, atoms, h=1e-5):
    f = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for k in range(3):
            e = []
            for s in (1, -1):
                b = atoms.copy()
                b.positions[i, k] += s * h
                e.append(energy(b))
            f[i, k] = -(e[0] - e[1]) / (2 * h)
    return f


def fd_stress(energy, atoms, h=1e-5):
    """(1/V) dE/d(strain), symmetrised; strain applied to cell and atoms together."""
    cell, frac, fd = atoms.cell.array.copy(), atoms.get_scaled_positions(), np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            e = []
            for s in (1, -1):
                eps = np.eye(3)
                eps[i, j] += s * h
                b = atoms.copy()
                b.set_cell(cell @ eps.T, scale_atoms=False)
                b.set_scaled_positions(frac)
                e.append(energy(b))
            fd[i, j] = (e[0] - e[1]) / (2 * h)
    return 0.5 * (fd + fd.T) / atoms.get_volume()


# ---------------------------------------------------------------- hypothesis
from hypothesis import HealthCheck, settings  # noqa: E402

# model compilation makes examples slow; keep example counts small and deterministic
settings.register_profile(
    "apax", max_examples=8, deadline=None, derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("apax")
