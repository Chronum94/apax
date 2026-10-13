import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk

from apax.config.model_config import EquivMPConfig, GMNNConfig, So3kratesConfig
from apax.data.preprocessing import compute_nl
from apax.layers.adaptive_cutoff import (
    _n_total,
    cutoff_bump,
    get_adaptive_cutoffs_solver,
    max_selected_count,
    pack_edges,
    safe_norm,
    select_edges,
    size_k_sel,
)
from apax.layers.properties import stress_times_vol
from apax.utils.convert import atoms_to_inputs

R_MAX = 6.0
NUM_NEIGHBORS = 12


def _cluster(seed=0):
    """Gas-phase cluster with a dense core and a sparse shell, so that the
    per-atom cutoffs differ substantially between atoms."""
    rng = np.random.default_rng(seed)
    core = rng.normal(scale=1.2, size=(14, 3))
    shell = rng.normal(size=(8, 3))
    shell = 4.0 * shell / np.linalg.norm(shell, axis=1, keepdims=True)
    R = np.concatenate([core, shell])
    Z = rng.choice([1, 6, 8], size=len(R))
    n = len(R)
    i, j = np.where(~np.eye(n, dtype=bool))  # all pairs: no NL boundary in FD
    return R, Z, np.stack([i, j]), np.zeros(3), np.zeros((len(i), 3))


def _periodic(seed=0):
    atoms = bulk("Cu", cubic=True).repeat((2, 2, 2))
    # triclinic, so that cell/stress transposition conventions matter
    shear = np.array([[1.0, 0.08, 0.0], [0.0, 1.0, 0.05], [0.03, 0.0, 1.06]])
    atoms.set_cell(atoms.cell.array @ shear, scale_atoms=True)
    atoms.rattle(0.15, seed=seed)
    atoms.positions[:8] *= 0.95  # local density variation
    inputs = atoms_to_inputs([atoms])
    R = inputs["positions"][0]
    box = inputs["box"][0]
    idx, offsets = compute_nl(R, box, R_MAX)
    return R, atoms.numbers, idx.astype(np.int32), box, offsets


def _dr(R, idx):
    return jnp.asarray(R)[idx[1]] - jnp.asarray(R)[idx[0]]


# -- selection --


def test_solver_root_and_invariances():
    R, Z, idx, *_ = _cluster()
    n = len(R)
    dr = _dr(R, idx)
    r = safe_norm(dr)
    mask = idx[0] != idx[1]
    cuts = get_adaptive_cutoffs_solver(idx[0], r, mask, NUM_NEIGHBORS, n, R_MAX, 0.5)

    res = _n_total(cuts, r, idx[0], mask, n, 0.5, 1 / R_MAX, NUM_NEIGHBORS)
    free = (cuts > R_MAX / 16 + 1e-6) & (cuts < R_MAX - 1e-6)
    assert np.all(np.abs(res - NUM_NEIGHBORS)[free] < 1e-4)
    assert np.ptp(np.asarray(cuts)) > 0.5  # core and shell differ

    # padding pairs (i == j) and edge order do not matter
    perm = np.random.default_rng(1).permutation(idx.shape[1])
    idx_p = np.concatenate([idx[:, perm], np.zeros((2, 9), int)], axis=1)
    dr_p = jnp.concatenate([dr[perm], jnp.zeros((9, 3))])
    a = select_edges(dr, idx, n, NUM_NEIGHBORS, R_MAX, 0.5)
    b = select_edges(dr_p, idx_p, n, NUM_NEIGHBORS, R_MAX, 0.5)
    np.testing.assert_allclose(a[0], b[0], atol=1e-12)
    np.testing.assert_array_equal(a[3], b[3])


def test_pack_and_overflow():
    R, Z, idx, *_ = _cluster()
    n = len(R)
    perm = np.random.default_rng(2).permutation(idx.shape[1])
    idx = np.concatenate([idx[:, perm], np.zeros((2, 5), int)], axis=1)
    dr = _dr(R, idx)
    _, pair_cutoffs, selected, counts = select_edges(
        dr, idx, n, NUM_NEIGHBORS, R_MAX, 0.5
    )
    w = cutoff_bump(safe_norm(dr), pair_cutoffs, 0.5)
    k = int(counts.max())

    dr_p, idx_p, w_p, overflow = pack_edges(dr, idx, selected, w, n, k)
    assert not overflow
    assert idx_p.shape == (2, n * k)
    real = np.asarray(idx_p[0] != idx_p[1])
    assert real.sum() == int(selected.sum())
    # rows are per center
    np.testing.assert_array_equal(
        np.asarray(idx_p[0])[real], np.repeat(np.arange(n), k)[real]
    )
    # padding slots: (0, 0), zero displacement, zero weight
    assert np.all(np.asarray(idx_p)[:, ~real] == 0)
    assert np.all(np.asarray(dr_p)[~real] == 0) and np.all(np.asarray(w_p)[~real] == 0)

    got = sorted(map(tuple, np.asarray(idx_p).T[real].tolist()))
    exp = sorted(map(tuple, idx.T[np.asarray(selected)].tolist()))
    assert got == exp
    np.testing.assert_allclose(w_p.sum(), (w * selected).sum())

    assert pack_edges(dr, idx, selected, w, n, k - 1)[3]


def test_max_selected_count_periodic():
    R, Z, idx, box, offsets = _periodic()
    config = _config(GMNNConfig)
    inputs = {"positions": [R], "numbers": [Z], "box": [box]}
    max_count = max_selected_count(config, inputs, [(idx, offsets)])
    assert NUM_NEIGHBORS - 4 <= max_count <= NUM_NEIGHBORS + 6
    assert size_k_sel(max_count, 4) == max_count + 4


# -- model level: shadow forces --


def _config(cls):
    return cls(
        nn=[16, 16],
        basis={"name": "bessel", "n_basis": 6, "r_max": R_MAX},
        descriptor_dtype="fp64",
        readout_dtype="fp64",
        adaptive_cutoff={"num_neighbors": NUM_NEIGHBORS},
        **(
            {"num_features": 16, "num_heads": 2, "max_degree": 2}
            if cls is So3kratesConfig
            else {}
        ),
        **({"features": 8, "num_iterations": 2} if cls is EquivMPConfig else {}),
    ).model_dump()


def _energy_fn(cls, inputs, k_sel, no_shadow=False):
    R, Z, idx, box, offsets = inputs
    model = cls.get_builder(None)(_config(cls)).build_energy_model(
        init_box=box, k_sel=k_sel
    )
    params = model.init(jax.random.PRNGKey(0), R, Z, idx, box, offsets)
    if no_shadow:
        model = model.clone(adaptive=dataclasses.replace(model.adaptive, no_shadow=True))

    def energy(R):
        return model.apply(params, R, Z, idx, box, offsets)[0]

    return energy


def _fd_grad(f, R, eps=1e-5):
    g = np.zeros_like(R)
    for a in range(R.shape[0]):
        for c in range(3):
            dR = np.zeros_like(R)
            dR[a, c] = eps
            g[a, c] = (f(R + dR) - f(R - dR)) / (2 * eps)
    return g


@pytest.mark.parametrize("cls", [GMNNConfig, EquivMPConfig, So3kratesConfig])
@pytest.mark.parametrize("system", [_cluster, _periodic])
def test_shadow_forces_match_finite_differences(cls, system):
    if cls is So3kratesConfig:
        pytest.importorskip("myrto")
    inputs = system()
    R = np.asarray(inputs[0], dtype=np.float64)
    k_sel = 40

    energy = _energy_fn(cls, inputs, k_sel)
    f = jax.jit(energy)
    grad = np.asarray(jax.jit(jax.grad(energy))(R))
    fd = _fd_grad(lambda x: float(f(x)), R)

    scale = np.abs(fd).max()
    assert scale > 0
    np.testing.assert_allclose(grad, fd, atol=2e-6 * scale)

    # Without the shadow term, forces are measurably wrong.
    energy_ns = _energy_fn(cls, inputs, k_sel, no_shadow=True)
    grad_ns = np.asarray(jax.jit(jax.grad(energy_ns))(R))
    np.testing.assert_allclose(energy_ns(R), energy(R))
    assert np.abs(grad_ns - fd).max() > 1e-3 * scale


@pytest.mark.parametrize("adaptive", [True, False])
def test_stress_matches_finite_differences(adaptive):
    """dE/d(eps) under homogeneous strain x -> x (1 + eps) of positions and cell
    (and thereby the periodic shifts), which is ASE's convention for stress * V."""
    R, Z, idx, box, offsets = _periodic()
    cell = box.T
    cls = GMNNConfig
    config = _config(cls)
    if not adaptive:
        config["adaptive_cutoff"] = None
    model = cls.get_builder(None)(config).build_energy_model(init_box=box, k_sel=40)
    params = model.init(jax.random.PRNGKey(0), R, Z, idx, box, offsets)

    def energy(R, box, perturbation=None, offsets=offsets):
        return model.apply(params, R, Z, idx, box, offsets, perturbation)[0]

    stress = np.asarray(stress_times_vol(energy, R, box))

    def strained(eps):
        F = np.eye(3) + eps
        return float(energy(R @ F, (cell @ F).T, offsets=offsets @ F))

    h = 1e-6
    fd = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            d = np.zeros((3, 3))
            d[a, b] = h
            fd[a, b] = (strained(d) - strained(-d)) / (2 * h)

    assert np.abs(fd).max() > 1e-3
    np.testing.assert_allclose(stress, fd, atol=1e-6 * np.abs(fd).max())


def test_energy_continuous_across_selection_boundary():
    inputs = _cluster()
    R = np.asarray(inputs[0])
    energy = jax.jit(_energy_fn(GMNNConfig, inputs, 40))
    # drag one shell atom radially through the region where it leaves/enters
    # neighbors' environments
    direction = R[-1] / np.linalg.norm(R[-1])
    ts = np.linspace(-1.5, 1.5, 301)
    es = []
    for t in ts:
        Rt = R.copy()
        Rt[-1] = R[-1] + t * direction
        es.append(float(energy(Rt)))
    es = np.array(es)
    d2 = np.abs(np.diff(es, 2))
    # no jumps: second differences stay at the level of smooth curvature
    assert d2.max() < 50 * np.median(d2) + 1e-8


def test_small_cell_self_images():
    """In a cell shorter than the cutoff, atoms see their own periodic images;
    selection must treat them like any other neighbor (same energy per atom for
    the primitive cell and a supercell)."""
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    atoms.rattle(0.05, seed=0)
    model = GMNNConfig.get_builder(None)(_config(GMNNConfig)).build_energy_model(
        init_box=np.eye(3), k_sel=60
    )

    def inputs(a):
        inp = atoms_to_inputs([a])
        R, box = inp["positions"][0], inp["box"][0]
        idx, offsets = compute_nl(R, box, R_MAX)
        return R, a.numbers, idx.astype(np.int32), box, offsets

    params = model.init(jax.random.PRNGKey(0), *inputs(atoms))
    small = inputs(atoms)
    assert np.any(small[2][0] == small[2][1])  # self-images present
    sc = atoms.repeat(3)
    e_small = model.apply(params, *small)[0] / len(atoms)
    e_sc = model.apply(params, *inputs(sc))[0] / len(sc)
    np.testing.assert_allclose(e_small, e_sc, atol=1e-10)
