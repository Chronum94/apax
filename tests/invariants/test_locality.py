"""Locality invariants of the short-range (GMNN) model, for any weights:

per-atom energy E_i depends only on atoms within r_max of i, F_i only on atoms within 2 r_max,
fragments farther apart than r_max are separable, isolated atoms give the element shift, and the
cutoff is smooth (E, F continuous, F -> 0 at r_max), with padding never producing NaN.
"""

import jax
import numpy as np
import pytest
from ase import Atoms
from hypothesis import given, settings
from hypothesis import strategies as st

from apax.data.input_pipeline import pad_nl
from apax.data.preprocessing import compute_nl
from apax.layers.scaling import PerElementScaleShift
from tests.invariants.conftest import (
    CELL_SHAPES,
    R_MAX,
    build,
    deformed_fcc,
    inputs,
    model_config,
    random_cluster,
)

CONFIGS = {
    "bessel": {},
    "gaussian-covalent": dict(
        basis={"name": "gaussian", "n_basis": 7, "r_max": R_MAX}, radial_transform="covalent"
    ),
    "corrections": dict(
        empirical_corrections=[
            {"name": "zbl"},
            {"name": "exponential"},
            {"name": "nlh", "r_max": R_MAX},
        ]
    ),
}


# ---------------------------------------------------------------- helpers (conftest candidates)
def rod(n=30, length=24.0, width=4.0, seed=0, species=(1, 6, 8)) -> Atoms:
    """Elongated cluster: many atoms beyond r_max and beyond 2 r_max of atom 0 (at one end)."""
    rng = np.random.default_rng(seed)
    pos = [np.zeros(3)]
    while len(pos) < n:
        p = rng.uniform(0, 1, 3) * (width, width, length)
        if all(np.linalg.norm(p - q) >= 1.0 for q in pos):
            pos.append(p)
    return Atoms(numbers=rng.choice(species, n), positions=pos)


def long_cell(seed=0) -> Atoms:
    """Periodic triclinic 1x1x8 fcc cell: thin in-plane (self-images), > 4 r_max along c."""
    atoms = deformed_fcc(CELL_SHAPES["triclinic"], (1, 1, 8))
    rng = np.random.default_rng(seed)
    atoms.numbers = rng.choice((29, 79), len(atoms))
    atoms.rattle(0.05, seed=seed)
    return atoms


SYSTEMS = {"cluster": rod, "periodic": long_cell}


def predict(builder, params, R, Z, idx, box, offsets):
    """Energy, forces, raw stress (dE/dF, not divided by V) and per-atom energies E_i, for explicit inputs."""
    model = builder.build_energy_derivative_model(init_box=box)
    out, state = model.apply(
        params, R, Z, idx, box, offsets,
        capture_intermediates=lambda m, _: isinstance(m, PerElementScaleShift),
    )
    res = {k: np.asarray(v) for k, v in out.items()}
    res["E_i"] = np.asarray(state["intermediates"]["energy_model"]["scale_shift"]["__call__"][0][:, 0])
    return res


def run(builder, params, atoms, cutoff=R_MAX):
    R, Z, _, box, _ = inputs(atoms)
    idx, offsets = compute_nl(R, box, cutoff)
    return predict(builder, params, R, Z, idx, box, offsets)


def mic_dist(atoms, i=0):
    return atoms.get_distances(i, range(len(atoms)), mic=True)


# ---------------------------------------------------------------- locality
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("system", SYSTEMS)
@settings(max_examples=2)
@given(seed=st.integers(0, 10_000))
def test_far_atoms_do_not_affect_atom_0(cfg, system, seed):
    """Moving / deleting atoms beyond r_max leaves E_0, beyond 2 r_max leaves F_0 unchanged (and
    moving atoms between r_max and 2 r_max does change F_0, so the check is not vacuous)."""
    atoms = SYSTEMS[system](seed=seed)
    builder, params = build(model_config(**CONFIGS[cfg]), atoms, seed=seed)
    ref = run(builder, params, atoms)
    d = mic_dist(atoms)
    rng = np.random.default_rng(seed)
    margin = 0.2  # > max displacement 0.1 * sqrt(3)

    def moved(mask):
        a = atoms.copy()
        a.positions[mask] += rng.uniform(-0.1, 0.1, (mask.sum(), 3))
        return run(builder, params, a)

    def deleted(mask):
        return run(builder, params, atoms[~mask])  # atom 0 is never deleted, keeps index 0

    beyond_1, beyond_2 = d > R_MAX + margin, d > 2 * R_MAX + margin
    shell = (d > R_MAX + margin) & (d < 2 * R_MAX - margin)
    assert beyond_2.any() and shell.any()
    for perturb in (moved, deleted):
        np.testing.assert_allclose(perturb(beyond_1)["E_i"][0], ref["E_i"][0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(perturb(beyond_2)["forces"][0], ref["forces"][0], rtol=0, atol=1e-12)
        assert np.abs(perturb(shell)["forces"][0] - ref["forces"][0]).max() > 1e-8


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("system", SYSTEMS)
def test_pairs_beyond_cutoff_contribute_nothing(cfg, system):
    """A neighbour list built to 2 r_max gives exactly the r_max result: the model's own cutoff
    removes the extra pairs, not just the neighbour list."""
    atoms = SYSTEMS[system](seed=1)
    builder, params = build(model_config(**CONFIGS[cfg]), atoms)
    ref, wide = run(builder, params, atoms), run(builder, params, atoms, cutoff=2 * R_MAX)
    for k in ("energy", "forces", "stress", "E_i"):
        np.testing.assert_allclose(wide[k], ref[k], rtol=0, atol=1e-10, err_msg=k)


# ---------------------------------------------------------------- separability
@pytest.mark.parametrize("cfg", CONFIGS)
@settings(max_examples=3)
@given(seed=st.integers(0, 10_000), periodic=st.booleans())
def test_fragments_beyond_cutoff_are_separable(cfg, seed, periodic):
    """E(A+B) = E(A) + E(B), forces/per-atom energies/virials split, also inside a large periodic box."""
    A = random_cluster(6, seed=seed)
    B = random_cluster(5, seed=seed + 1)
    B.positions += (6.0 + R_MAX + 0.5, 0, 0)  # clusters live in [0, 6]^3
    AB = A + B
    if periodic:
        AB.cell, AB.pbc = np.diag([AB.positions[:, 0].max() + R_MAX + 0.5, 6 + R_MAX + 0.5, 6 + R_MAX + 0.5]), True
    builder, params = build(model_config(**CONFIGS[cfg]), A, seed=seed)
    a, b, ab = (run(builder, params, x) for x in (A, B, AB))
    np.testing.assert_allclose(ab["energy"], a["energy"] + b["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ab["forces"], np.concatenate([a["forces"], b["forces"]]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(ab["E_i"], np.concatenate([a["E_i"], b["E_i"]]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(ab["stress"], a["stress"] + b["stress"], rtol=0, atol=1e-10)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("activation", ["swish", "elu", "identity"])
@pytest.mark.parametrize("periodic", [False, True])
def test_isolated_atom_energy_is_element_shift(cfg, activation, periodic):
    Z = [1, 6, 8, 29, 79]
    atoms = Atoms(numbers=Z, positions=np.arange(5)[:, None] * [R_MAX + 1, 0, 0])
    if periodic:
        atoms.cell, atoms.pbc = np.diag([5 * (R_MAX + 1), R_MAX + 1, R_MAX + 1]), True
    builder, params = build(model_config(readout_activation=activation, **CONFIGS[cfg]), atoms)
    shift = np.random.default_rng(0).normal(size=(119, 1))
    params["params"]["energy_model"]["scale_shift"]["shift_per_element"] = shift
    res = run(builder, params, atoms)
    np.testing.assert_allclose(res["E_i"], shift[Z, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(res["energy"], shift[Z, 0].sum(), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(res["forces"], 0.0)
    np.testing.assert_array_equal(res["stress"], 0.0)


# ---------------------------------------------------------------- cutoff smoothness
def dimer_scan(cfg, periodic, species=(6, 8), d_grid=None):
    atoms = Atoms(numbers=species, positions=np.zeros((2, 3)))
    if periodic:
        atoms.cell, atoms.pbc = np.diag([3 * R_MAX, 2.5 * R_MAX, 2.5 * R_MAX]), True
    builder, params = build(model_config(**CONFIGS[cfg]), atoms)
    apply = jax.jit(builder.build_energy_derivative_model(init_box=inputs(atoms)[3]).apply)
    E, F = [], []
    for d in d_grid:
        a = atoms.copy()
        a.positions[1, 0] = d
        R, Z, idx, box, offsets = inputs(a)
        out = apply(params, R, Z, idx, box, offsets)
        E.append(float(out["energy"]))
        F.append(np.asarray(out["forces"])[1, 0])
        assert all(np.isfinite(v).all() for v in out.values()), d
    return np.array(E), np.array(F)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("periodic", [False, True])
def test_dimer_energy_and_force_are_smooth_through_cutoff(cfg, periodic):
    h = 1e-3
    d = np.concatenate([np.arange(R_MAX - 0.3, R_MAX + 0.3, h), [R_MAX - 1e-6]])
    E, F = dimer_scan(cfg, periodic, d_grid=d)
    E_far, inside = E[d >= R_MAX][0], d < R_MAX
    # beyond r_max: exactly the isolated-atoms result
    np.testing.assert_allclose(E[~inside], E_far, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(F[~inside], 0.0)
    # E continuous: no jump larger than the force allows (mean value theorem) on the fine grid
    grid = slice(0, -1)
    dE, Fmax = np.abs(np.diff(E[grid])), np.maximum(np.abs(F[grid][1:]), np.abs(F[grid][:-1]))
    assert np.all(dE <= 1.1 * Fmax * h + 1e-12), dE.max()
    # F continuous and -> 0 at r_max
    scale = np.abs(F).max()
    assert scale > 1e-6
    assert np.abs(np.diff(F[grid])).max() < 0.05 * scale
    assert abs(F[-1]) < 1e-4 * scale and abs(E[-1] - E_far) < 1e-8 * scale


@pytest.mark.parametrize("periodic", [False, True])
def test_padded_atoms_and_pairs_change_nothing(periodic):
    """Padding as the input pipeline does it: zero-length (0, 0) pairs and Z = 0 atoms at R = 0."""
    atoms = long_cell(seed=2) if periodic else random_cluster(10, seed=2)
    builder, params = build(model_config(), atoms)
    R, Z, idx, box, offsets = inputs(atoms)
    ref = predict(builder, params, R, Z, idx, box, offsets)
    idx_p, offsets_p = pad_nl(idx, offsets, idx.shape[1] + 7)
    R_p, Z_p = np.pad(R, ((0, 3), (0, 0))), np.pad(Z, (0, 3))
    pad = predict(builder, params, R_p, Z_p, idx_p, box, offsets_p)
    assert all(np.isfinite(v).all() for v in pad.values())
    n = len(atoms)
    np.testing.assert_allclose(pad["energy"], ref["energy"], rtol=1e-12)
    np.testing.assert_allclose(pad["forces"][:n], ref["forces"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(pad["forces"][n:], 0.0)
    np.testing.assert_allclose(pad["stress"], ref["stress"], rtol=0, atol=1e-12)
