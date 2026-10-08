"""Symmetry invariants that hold for ANY weights: Euclidean group, permutation, cell representation, extensivity.

Every check runs on both the training path (offset_path) and the real ASECalculator.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule, niggli_reduce
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from apax.config.model_config import EquivMPConfig
from tests.invariants.conftest import (
    CELL_SHAPES,
    R_MAX,
    ase_calculator,
    ase_eval,
    build,
    model_config,
    offset_path,
    random_bulk,
    random_cluster,
)

PATHS = ["offset", "ase"]
TOL = dict(rtol=1e-9, atol=1e-9)

_MODELS = {}


def model(name):
    """(cfg, builder, params) per model, built once. ponytail: module cache instead of a session fixture."""
    if name not in _MODELS:
        if name == "gmnn":
            cfg = model_config()
        else:
            cfg = EquivMPConfig(
                basis={"name": "bessel", "n_basis": 8, "r_max": R_MAX}, nn=[16, 16], features=8, max_degree=2,
                calc_stress=True, descriptor_dtype="fp64", readout_dtype="fp64",
            )
        _MODELS[name] = (cfg, *build(cfg, random_bulk(), seed=1))
    return _MODELS[name]


def predict(path, atoms, name="gmnn"):
    cfg, builder, params = model(name)
    pbc = atoms.pbc.all()
    if path == "offset":
        r = offset_path(builder, params, atoms)
        r = {"energy": float(r["energy"]), "forces": r["forces"], **({"stress": r["stress"]} if pbc else {})}
    else:
        r = ase_eval(ase_calculator(cfg, params), atoms, ("energy", "forces", "stress") if pbc else ())
    return r


def assert_same(a, b, F_map=lambda f: f, S_map=lambda s: s, n=1):
    np.testing.assert_allclose(b["energy"], n * a["energy"], **TOL)
    np.testing.assert_allclose(b["forces"], F_map(a["forces"]), **TOL)
    if "stress" in a:
        np.testing.assert_allclose(b["stress"], S_map(a["stress"]), **TOL)


def systems():
    """Clusters plus thin (rep=2, height < 2 r_max) and thick (rep=3) cells of several shapes."""
    out = {"cluster": random_cluster(n=14, seed=3)}
    for shape in ("cubic", "monoclinic", "triclinic", "skewed"):
        out[f"{shape}-thin"] = random_bulk(shape, rep=2, seed=4)
    for shape in ("orthorhombic", "triclinic"):
        out[f"{shape}-thick"] = random_bulk(shape, rep=3, seed=5)
    out["prim-fcc"] = random_bulk_prim()
    return out


def random_bulk_prim(seed=0):
    """Two-atom primitive-fcc-based cell, far thinner than r_max (many self images)."""
    atoms = bulk("Cu", "fcc", a=3.6).repeat((2, 1, 1))
    atoms.numbers = [29, 79]
    atoms.rattle(0.05, seed=seed)
    return atoms


SYSTEMS = systems()
PBC_SYSTEMS = [k for k in SYSTEMS if k != "cluster"]


@pytest.fixture(params=PATHS)
def path(request):
    return request.param


# ---------------------------------------------------------------- translation
@pytest.mark.parametrize("name", list(SYSTEMS))
def test_rigid_translation(path, name):
    atoms = SYSTEMS[name]
    moved = atoms.copy()
    moved.positions += [0.37, -1.21, 2.9]
    assert_same(predict(path, atoms), predict(path, moved))


@pytest.mark.parametrize("name", PBC_SYSTEMS)
def test_wrap_and_lattice_shifts(path, name):
    """Wrapping into the cell or moving single atoms by lattice vectors changes nothing."""
    atoms = SYSTEMS[name]
    ref = predict(path, atoms)
    rng = np.random.default_rng(0)
    shifted = atoms.copy()
    n = rng.integers(-2, 3, (len(atoms), 3))
    shifted.positions += n @ atoms.cell.array
    assert_same(ref, predict(path, shifted))
    shifted.wrap()
    assert_same(ref, predict(path, shifted))


# ---------------------------------------------------------------- rotation / inversion
@pytest.mark.parametrize("model_name", ["gmnn", "equiv-mp"])
@pytest.mark.parametrize("name", ["cluster", "triclinic-thin", "prim-fcc"])
@given(seed=st.integers(0, 2**16), improper=st.booleans())
def test_rotation_and_inversion(path, name, model_name, seed, improper):
    atoms = SYSTEMS[name]
    Q = Rotation.random(random_state=seed).as_matrix() * (-1 if improper else 1)
    rot = atoms.copy()
    rot.positions = atoms.positions @ Q.T
    if atoms.pbc.all():
        rot.set_cell(atoms.cell.array @ Q.T, scale_atoms=False)
    assert_same(
        predict(path, atoms, model_name), predict(path, rot, model_name),
        F_map=lambda f: f @ Q.T, S_map=lambda s: Q @ s @ Q.T,
    )


@pytest.mark.parametrize("name", ["cluster", "skewed-thin"])
def test_pure_inversion(path, name):
    atoms = SYSTEMS[name]
    inv = atoms.copy()
    inv.positions = -atoms.positions
    if atoms.pbc.all():
        inv.set_cell(-atoms.cell.array, scale_atoms=False)
    assert_same(predict(path, atoms), predict(path, inv), F_map=lambda f: -f)


# ---------------------------------------------------------------- permutation
@pytest.mark.parametrize("name", ["cluster", "triclinic-thin", "prim-fcc"])
def test_permutation(path, name):
    atoms = SYSTEMS[name]
    perm = np.random.default_rng(1).permutation(len(atoms))
    assert_same(predict(path, atoms), predict(path, atoms[perm]), F_map=lambda f: f[perm])


# ---------------------------------------------------------------- cell representation
UNIMODULAR = {
    "a+b": [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
    "mixed": [[1, 0, -1], [2, 1, 0], [0, 1, 1]],  # det 1
    "swap": [[0, 1, 0], [1, 0, 0], [0, 0, 1]],  # det -1 (left-handed basis)
}


@pytest.mark.parametrize("M", list(UNIMODULAR))
@pytest.mark.parametrize("name", ["cubic-thin", "triclinic-thick", "prim-fcc"])
def test_unimodular_cell_change(path, name, M):
    atoms = SYSTEMS[name]
    alt = atoms.copy()
    alt.set_cell(np.asarray(UNIMODULAR[M]) @ atoms.cell.array, scale_atoms=False)
    alt.wrap()
    assert_same(predict(path, atoms), predict(path, alt))


@pytest.mark.parametrize("name", ["skewed-thin", "prim-fcc"])
def test_niggli_reduced(path, name):
    atoms = SYSTEMS[name].copy()
    atoms.set_cell(np.asarray(UNIMODULAR["mixed"]) @ atoms.cell.array, scale_atoms=False)  # make it non-reduced
    red = atoms.copy()
    niggli_reduce(red)  # keeps atom order, may rotate the cell: compare in the reduced frame
    Q = np.linalg.solve(atoms.cell.array, red.cell.array)  # not orthogonal in general; derive rotation from positions
    Q = _rotation_between(atoms, red)
    assert_same(predict(path, atoms), predict(path, red), F_map=lambda f: f @ Q.T, S_map=lambda s: Q @ s @ Q.T)


def _rotation_between(a, b):
    """Orthogonal Q with b.positions - b0 ≈ (a.positions - a0) @ Q.T modulo lattice: from the Gram-preserving cell map."""
    # same lattice, so b.cell = U @ a.cell @ Q.T for integer U; solve for Q via polar decomposition of a.cell^-1 U^-1 b.cell
    U = np.rint(b.cell.array @ np.linalg.inv(a.cell.array)).astype(float)
    X = np.linalg.solve(U @ a.cell.array, b.cell.array)  # = Q.T
    if np.allclose(X @ X.T, np.eye(3)):
        return X.T
    u, _, vt = np.linalg.svd(X)
    return (u @ vt).T


# ---------------------------------------------------------------- extensivity
@pytest.mark.parametrize("rep", [(2, 1, 1), (1, 2, 3)])
@pytest.mark.parametrize("name", ["triclinic-thin", "prim-fcc"])
def test_supercell(path, name, rep):
    atoms = SYSTEMS[name]
    n = int(np.prod(rep))
    assert_same(
        predict(path, atoms), predict(path, atoms.repeat(rep)),
        F_map=lambda f: np.tile(f, (n, 1)), n=n,
    )


# ---------------------------------------------------------------- huge box == gas phase
@pytest.mark.parametrize("model_name", ["gmnn", "equiv-mp"])
@pytest.mark.parametrize("box", [30.0, 4 * R_MAX + 1.0])
def test_molecule_in_huge_box(path, model_name, box):
    gas = molecule("CH3CH2OH")
    gas.positions += 3.0  # off-centre, some atoms close to a box face
    solid = gas.copy()
    solid.set_cell(np.diag([box, box * 1.1, box * 0.95]) @ np.array(CELL_SHAPES["triclinic"]).T, scale_atoms=False)
    solid.pbc = True
    a, b = predict(path, gas, model_name), predict(path, solid, model_name)
    np.testing.assert_allclose(b["energy"], a["energy"], **TOL)
    np.testing.assert_allclose(b["forces"], a["forces"], **TOL)
