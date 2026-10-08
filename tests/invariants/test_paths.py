"""Same params + same structure → same E/F/stress through every route apax offers.

Reference is the training path (`offset_path`). Routes: ASECalculator (jax_md NL, jax_md without cell list,
vesin), ASECalculator.batch_eval, the MD energy/auxiliary functions from `md_setup`, padded inputs,
a reused calculator over changing structures, unit conversion of labels, and matmul precision.
"""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.units import Bohr, kcal, mol

from apax.config.md_config import MDConfig
from apax.md.simulate import md_setup
from apax.train.checkpoints import canonicalize_energy_model_parameters
from apax.utils.convert import atoms_to_inputs, atoms_to_labels
from tests.invariants.conftest import (
    ase_calculator,
    ase_eval,
    build,
    inputs,
    model_config,
    offset_path,
    random_bulk,
    random_cluster,
)

PROPS = ("energy", "forces", "stress")
TOL = dict(rtol=1e-10, atol=1e-10)

STRUCTURES = {
    "cluster": lambda: random_cluster(n=12, seed=1),
    "thin-triclinic": lambda: random_bulk("triclinic", rep=2, seed=2),  # → vesin
    "thick-monoclinic": lambda: random_bulk("monoclinic", rep=3, seed=3),  # → jax_md
}


def assert_same(got, ref):
    assert {"energy", "forces"} <= set(got)
    for k in set(got) & set(PROPS):
            np.testing.assert_allclose(np.asarray(got[k]), ref[k], **TOL, err_msg=k)


@pytest.fixture(scope="module")
def setup():
    """One model + reference predictions per structure (built once, shared by the route tests)."""
    cfg = model_config()
    out = {}
    with jax.enable_x64(True):
        for name, make in STRUCTURES.items():
            atoms = make()
            builder, params = build(cfg, atoms)
            out[name] = (cfg, params, atoms, offset_path(builder, params, atoms))
    return out


# ---------------------------------------------------------------- ASECalculator routes
@pytest.mark.parametrize("name", STRUCTURES)
@pytest.mark.parametrize("disable_cell_list", [False, True])
def test_ase_calculator(setup, name, disable_cell_list):
    cfg, params, atoms, ref = setup[name]
    calc = ase_calculator(cfg, params, disable_cell_list=disable_cell_list)
    props = PROPS if atoms.pbc.all() else ("energy", "forces")
    assert_same(ase_eval(calc, atoms, props), ref)
    assert calc.neigbor_from_jax == (name != "thin-triclinic")  # the route we meant to hit


@pytest.mark.parametrize("name", STRUCTURES)
def test_batch_eval(setup, name):
    cfg, params, atoms, ref = setup[name]
    calc = ase_calculator(cfg, params)
    # a smaller second structure in the same batch forces atom + neighbour padding of the first
    small = atoms[: len(atoms) // 2]
    (a, _) = calc.batch_eval([atoms, small], batch_size=2, silent=True)
    got = {"energy": a.get_potential_energy(), "forces": a.get_forces()}
    if atoms.pbc.all():
        got["stress"] = a.get_stress(voigt=False)
    assert_same(got, ref)


# ---------------------------------------------------------------- MD route
@pytest.mark.parametrize("name", ["cluster", "thick-monoclinic"])
def test_md_functions(setup, name, tmp_path):
    """The energy the integrator differentiates and the auxiliary (logged) predictions."""
    cfg, params, atoms, ref = setup[name]
    write(tmp_path / "init.extxyz", atoms)
    md_config = MDConfig(duration=1.0, initial_structure=str(tmp_path / "init.extxyz"), properties=list(PROPS))
    config = MagicMock()
    config.model = cfg
    with patch("apax.md.simulate.restore_parameters", return_value=(config, params)):
        system, fns = md_setup(config, md_config)

    nbr = fns.neighbor_fn.allocate(system.positions)
    R, Z, box = system.positions, system.atomic_numbers, system.box
    aux = fns.auxiliary_fn(R, Z, nbr.idx, box, jnp.zeros((nbr.idx.shape[1], 3)))
    if atoms.pbc.all():
        aux["stress"] = aux["stress"] / atoms.get_volume()
    assert_same(aux, ref)

    # jax_md integrators use -grad(energy_fn); with fractional coordinates jax_md's transform makes this the
    # real-space force
    E, dE = jax.value_and_grad(lambda R: fns.energy_fn(R, neighbor=nbr.idx, box=box))(R)
    assert_same({"energy": E, "forces": -dE}, ref)


# ---------------------------------------------------------------- padding
@pytest.mark.parametrize("name", STRUCTURES)
def test_padding_with_dummy_atoms_and_empty_neighbours(setup, name):
    cfg, params, atoms, ref = setup[name]
    R, Z, idx, box, offsets = inputs(atoms)
    n_pad_atoms, n_pad_pairs = 5, 37
    R = np.pad(R, ((0, n_pad_atoms), (0, 0)))  # pipeline convention: zeros
    Z = np.pad(Z, (0, n_pad_atoms))
    idx = np.pad(idx, ((0, 0), (0, n_pad_pairs)))  # (0, 0) pairs, zero offset
    offsets = np.pad(offsets, ((0, n_pad_pairs), (0, 0)))
    builder = cfg.get_builder()(cfg.model_dump())
    out = builder.build_energy_derivative_model(init_box=box).apply(params, R, Z, idx, box, offsets)
    got = {k: np.asarray(v) for k, v in out.items()}
    np.testing.assert_array_equal(got["forces"][len(atoms) :], 0.0)
    got["forces"] = got["forces"][: len(atoms)]
    if atoms.pbc.all():
        got["stress"] = got["stress"] / atoms.get_volume()
    assert_same(got, ref)


# ---------------------------------------------------------------- calculator state
def _transitions():
    """Consecutive structures, each a different kind of change from the previous one."""
    base = random_bulk("triclinic", rep=2, seed=4)  # vesin
    fewer = base[:-3]  # atom count
    species = fewer.copy()
    species.numbers = np.roll(species.numbers, 1)  # species (same count)
    species.numbers[0] = 47
    cell = species.copy()  # cell only, stays on vesin
    cell.set_cell(cell.cell.array * 1.03, scale_atoms=True)
    gas = random_cluster(n=len(cell), seed=5)  # pbc on → off (jax_md free)
    gas2 = random_cluster(n=len(cell), seed=6)  # positions only, gas phase
    back = random_bulk("cubic", rep=2, seed=7)  # pbc off → on
    thick = random_bulk("monoclinic", rep=3, seed=8)  # vesin → jax_md
    thin_again = random_bulk("skewed", rep=2, seed=9)  # jax_md → vesin
    return [base, fewer, species, cell, gas, gas2, back, thick, thin_again]


def test_reused_calculator_equals_fresh_one():
    cfg = model_config()
    seq = _transitions()
    with jax.enable_x64(True):
        _, params = build(cfg, seq[0], scale=30.0)
        reused = ase_calculator(cfg, params)
        for i, atoms in enumerate(seq):
            props = PROPS if atoms.pbc.all() else ("energy", "forces")
            got = ase_eval(reused, atoms, props)
            ref = ase_eval(ase_calculator(cfg, params), atoms, props)
            for k in props:
                np.testing.assert_allclose(got[k], ref[k], **TOL, err_msg=f"step {i}: {k}")


# ---------------------------------------------------------------- units
def test_units_kcal_bohr_equal_ev_ang():
    """Same physical data written in kcal/mol + Bohr gives the same internal (eV, Å) inputs and labels."""
    rng = np.random.default_rng(0)
    ev_ang = random_bulk("triclinic", rep=2, seed=10)
    E, F, S = rng.normal(), rng.normal(size=(len(ev_ang), 3)), rng.normal(size=6)
    ev_ang.calc = SinglePointCalculator(ev_ang, energy=E, forces=F, stress=S)

    k = kcal / mol  # eV per kcal/mol
    kcal_bohr = ev_ang.copy()
    kcal_bohr.set_cell(ev_ang.cell.array / Bohr, scale_atoms=True)
    kcal_bohr.calc = SinglePointCalculator(kcal_bohr, energy=E / k, forces=F / k * Bohr, stress=S / k * Bohr**3)

    a_inp = atoms_to_inputs([ev_ang.copy()])
    b_inp = atoms_to_inputs([kcal_bohr.copy()], pos_unit="Bohr")
    for key in ("positions", "box"):
        np.testing.assert_allclose(b_inp[key][0], a_inp[key][0], rtol=1e-12, atol=1e-12, err_msg=key)

    a_lab = atoms_to_labels([ev_ang], pos_unit="Ang", energy_unit="eV")
    b_lab = atoms_to_labels([kcal_bohr], pos_unit="Bohr", energy_unit="kcal/mol")
    for key in ("energy", "forces", "stress"):
        np.testing.assert_allclose(b_lab[key][0], a_lab[key][0], rtol=1e-12, atol=1e-12, err_msg=key)


# ---------------------------------------------------------------- precision
def _fp32_config():
    return model_config(descriptor_dtype="fp32", readout_dtype="fp32")  # scale_shift stays fp64, as in apax defaults


def test_no_default_precision_dot_in_fp32_model():
    """On Ampere+ GPUs, an fp32 dot with DEFAULT precision runs in TF32 (~1e-3 relative). Every dot_general in
    the jitted energy+forces+stress model must request HIGHEST."""
    cfg = _fp32_config()
    atoms = random_bulk("triclinic", rep=2, seed=11)
    builder, params = build(cfg, atoms)
    inp = inputs(atoms)
    model = builder.build_energy_derivative_model(init_box=inp[3])
    text = jax.jit(model.apply).lower(params, *inp).as_text()
    dots = [line for line in text.splitlines() if "dot_general" in line]
    assert dots, "no dot_general found: the check would be vacuous"
    bad = [d.strip()[:160] for d in dots if "HIGHEST" not in d]
    assert not bad, f"{len(bad)}/{len(dots)} dot_general ops without HIGHEST precision, e.g. {bad[:3]}"


@pytest.mark.parametrize("name", STRUCTURES)
def test_fp32_model_matches_fp64(setup, name):
    """Same weights in fp32: E and F within single-precision round-off of the fp64 model (O(1) outputs)."""
    _, params, atoms, ref = setup[name]
    cfg = _fp32_config()
    builder = cfg.get_builder()(cfg.model_dump())
    inp = inputs(atoms)
    params = jax.tree_util.tree_map(lambda x: x.astype(np.float32) if x.dtype == np.float64 else x, params)
    out = builder.build_energy_derivative_model(init_box=inp[3]).apply(params, *inp)
    # ponytail: 1e-4 relative to the force scale is ~1000 fp32 ulps; TF32 or a dropped term would be far worse
    f_scale = np.abs(ref["forces"]).max()
    np.testing.assert_allclose(out["energy"], ref["energy"], rtol=1e-5, atol=1e-5 * len(atoms))
    np.testing.assert_allclose(out["forces"], ref["forces"], atol=1e-4 * f_scale)
