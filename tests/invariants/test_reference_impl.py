"""Naive numpy reference implementations vs apax.

Descriptor, neighbour lists, empirical repulsions and property heads, each recomputed from the formulas with
plain loops and compared to apax at fp64 precision.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from jax_md import partition, space
from scipy.spatial.transform import Rotation

from apax.data.preprocessing import compute_nl
from apax.layers.empirical import ExponentialRepulsion, ZBLRepulsion
from tests.invariants.conftest import (
    CELL_SHAPES,
    R_MAX,
    build,
    deformed_fcc,
    inputs,
    model_config,
    offset_path,
    random_bulk,
    random_cluster,
)


# ---------------------------------------------------------------- reference neighbour list
def brute_force_pairs(atoms, r_max=R_MAX):
    """All (i, j, v) with v = r_j + S·cell - r_i, 0 < |v| < r_max, over every image S that can reach."""
    pos = atoms.positions
    if not atoms.pbc.any():
        shifts = [np.zeros(3)]
    else:
        cell = atoms.cell.array
        vol = abs(np.linalg.det(cell))
        heights = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
        n = [int(np.ceil(r_max / h)) + 1 for h in heights]
        shifts = [np.array(S) @ cell for S in itertools.product(*(range(-m, m + 1) for m in n))]
    pairs = []
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            for s in shifts:
                v = pos[j] + s - pos[i]
                d = np.linalg.norm(v)
                if 1e-8 < d < r_max:
                    pairs.append((i, j, v))
    return pairs


def canonical(pairs):
    """Sort (i, j, v) so two lists describing the same pair set line up element by element."""
    pairs = sorted(pairs, key=lambda p: (p[0], p[1], *np.round(p[2], 4)))
    return np.array([p[:2] for p in pairs]), np.array([p[2] for p in pairs])


def assert_same_pairs(a, b):
    ij_a, v_a = canonical(a)
    ij_b, v_b = canonical(b)
    assert len(ij_a) == len(ij_b)
    np.testing.assert_array_equal(ij_a, ij_b)
    np.testing.assert_allclose(v_a, v_b, atol=1e-10)


def apax_pairs(atoms):
    R, _, idx, box, offsets = inputs(atoms)
    if atoms.pbc.any():
        vec = (R[idx[1]] - R[idx[0]]) @ box.T + offsets
    else:
        vec = R[idx[1]] - R[idx[0]]
    return list(zip(idx[0], idx[1], vec))


def jax_md_pairs(atoms, r_max=R_MAX):
    """Exactly as ASECalculator builds the jax_md neighbour list (fractional coordinates, Sparse)."""
    if atoms.pbc.any():
        box = jnp.asarray(atoms.cell.array.T)
        disp, _ = space.periodic_general(box, fractional_coordinates=True)
        R = jnp.asarray(atoms.get_scaled_positions())
        nl_fn = partition.neighbor_list(disp, box, r_max, 0.5, fractional_coordinates=True, format=partition.Sparse)
    else:
        disp, _ = space.free()
        R = jnp.asarray(atoms.positions)
        nl_fn = partition.neighbor_list(
            disp, jnp.zeros((3, 3)), r_max, 0.5, fractional_coordinates=True,
            disable_cell_list=True, format=partition.Sparse,
        )
    idx = np.asarray(nl_fn.allocate(R).idx)
    idx = idx[:, idx[0] < len(atoms)]
    vec = np.asarray(jax.vmap(disp)(R[idx[1]], R[idx[0]]))
    keep = np.linalg.norm(vec, axis=1) < r_max  # jax_md keeps r_max + skin
    return list(zip(idx[0][keep], idx[1][keep], vec[keep]))


def thin_cells():
    """Cells with lattice planes closer than r_max: single atoms see many of their own images."""
    yield "fcc_primitive", bulk("Cu", "fcc", a=3.6)
    yield "bcc_primitive_triclinic", Atoms("Fe", cell=[[2.5, 0, 0], [0.8, 2.6, 0], [0.5, 0.7, 2.4]], pbc=True)
    yield "slab_like", deformed_fcc(CELL_SHAPES["monoclinic"], (1, 1, 4))
    for shape in CELL_SHAPES:
        yield f"{shape}_rep1", random_bulk(shape, rep=1, seed=1)


NL_CASES = [*thin_cells()] + [(f"{s}_rep2", random_bulk(s, rep=2, seed=2)) for s in CELL_SHAPES]
NL_CASES += [("cluster", random_cluster(20, seed=3))]


@pytest.mark.parametrize("atoms", [a for _, a in NL_CASES], ids=[n for n, _ in NL_CASES])
def test_compute_nl_matches_brute_force(atoms):
    assert_same_pairs(apax_pairs(atoms), brute_force_pairs(atoms))


JAX_MD_CASES = [(s, random_bulk(s, rep=3, seed=4)) for s in ("cubic", "tetragonal", "orthorhombic")]
JAX_MD_CASES += [("cluster", random_cluster(20, seed=5))]


@pytest.mark.parametrize("atoms", [a for _, a in JAX_MD_CASES], ids=[n for n, _ in JAX_MD_CASES])
def test_jax_md_nl_matches_brute_force(atoms):
    # only cells with min height > 2 r_max, the regime where ASECalculator uses jax_md
    from apax.md.ase_calc import neighbor_calculable_with_jax

    assert neighbor_calculable_with_jax(atoms.cell.array, R_MAX)
    assert_same_pairs(jax_md_pairs(atoms), brute_force_pairs(atoms))


# ---------------------------------------------------------------- reference Gaussian-moment descriptor
def ref_basis(r, basis):
    n_b, rc = basis["n_basis"], basis["r_max"]
    k = np.arange(n_b)
    if basis["name"] == "bessel":
        x1, x2 = np.pi * (k + 1) * r / rc, np.pi * (k + 2) * r / rc
        return (-1.0) ** k * np.sqrt(2) * np.pi / rc**1.5 * (np.sin(x1) / x1 + np.sin(x2) / x2)
    beta = n_b**2 / rc**2
    shifts = basis["r_min"] + (rc - basis["r_min"]) / n_b * k
    # the 1/sqrt(n_basis) embedding norm is applied to the Gaussian basis only
    return (2 * beta / np.pi) ** 0.25 * np.exp(-beta * (shifts - r) ** 2) / np.sqrt(n_b)


def tril2(n):
    return [(r, s) for r in range(n) for s in range(r, n)]


def tril3(n):
    # apax's ordering: (i,i,i), then (i,j,j) for every j != i, then (i,j,k) with i < j < k
    out = []
    for i in range(n):
        out.append((i, i, i))
        out += [(i, j, j) for j in range(n) if j != i]
        out += [(i, j, k) for j in range(i + 1, n) for k in range(j + 1, n)]
    return out


def ref_descriptor(atoms, emb, basis, n_radial):
    """Per-atom Gaussian moments with explicit loops; emb[Z_centre, Z_neighbour] are the pair coefficients."""
    rc, Z, n = basis["r_max"], atoms.numbers, n_radial
    M = [np.zeros((len(atoms), n) + (3,) * k) for k in range(4)]
    for i, j, v in brute_force_pairs(atoms, rc):
        # centre j, neighbour i, v points neighbour -> centre (sign is irrelevant: all contractions are even)
        r = np.linalg.norm(v)
        u = v / (r + 1e-5)  # ponytail: apax's regularised unit vector, kept verbatim
        g = emb[Z[j], Z[i]] @ ref_basis(r, basis) * 0.5 * (np.cos(np.pi * r / rc) + 1)
        for a in range(n):
            M[0][j, a] += g[a]
            M[1][j, a] += g[a] * u
            M[2][j, a] += g[a] * np.outer(u, u)
            M[3][j, a] += g[a] * np.einsum("i,j,k->ijk", u, u, u)
    feats = []
    for m0, m1, m2, m3 in zip(*M):
        m3f = m3.reshape(n, 9, 3)  # (radial, ij, k)
        c = [list(m0)]
        c.append([m1[r] @ m1[s] for r, s in tril2(n)])
        c.append([np.sum(m2[r] * m2[s]) for r, s in tril2(n)])
        c.append([np.sum(m3[r] * m3[s]) for r, s in tril2(n)])
        c.append([np.trace(m2[r] @ m2[s] @ m2[t]) for r, s, t in tril3(n)])
        c.append([m1[r] @ m2[t] @ m1[s] for r, s in tril2(n) for t in range(n)])
        c.append([np.sum((m3f[r].T @ m3f[s]) * m2[t]) for r, s in tril2(n) for t in range(n)])
        c.append([m3f[r].T @ m2[s].ravel() @ m1[t] for r in range(n) for s in range(n) for t in range(n)])
        feats.append(np.concatenate(c))
    return np.array(feats)


BASES = {
    "bessel": {"name": "bessel", "n_basis": 6, "r_max": R_MAX},
    "gaussian": {"name": "gaussian", "n_basis": 6, "r_min": 0.5, "r_max": R_MAX, "spacing": "linear"},
}
DESC_CASES = {
    "cluster": random_cluster(10, seed=6),
    "triclinic_rep2": random_bulk("triclinic", rep=2, seed=7),
    "fcc_primitive": bulk("Cu", "fcc", a=3.6),
}


@pytest.mark.parametrize("basis", BASES.values(), ids=BASES.keys())
@pytest.mark.parametrize("atoms", DESC_CASES.values(), ids=DESC_CASES.keys())
def test_descriptor_matches_reference(atoms, basis):
    n_radial = 4
    cfg = model_config(basis=basis, n_radial=n_radial, nn=[8])
    builder = cfg.get_builder()(cfg.model_dump())
    inp = inputs(atoms)
    model = builder.build_feature_model(only_use_n_layers=0, init_box=inp[3], should_average=False)
    params = model.init(jax.random.PRNGKey(0), *inp)
    g = np.asarray(model.apply(params, *inp))

    emb = np.asarray(params["params"]["representation"]["radial_fn"]["atomic_type_embedding"])
    ref = ref_descriptor(atoms, emb, basis, n_radial)
    assert g.shape == ref.shape
    np.testing.assert_allclose(g, ref, rtol=1e-9, atol=1e-10 * np.abs(ref).max())


# ---------------------------------------------------------------- empirical repulsions
ZBL_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
ZBL_E = np.array([3.19980, 0.94229, 0.4029, 0.20162])


def zbl_pair(r, z1, z2, rc):
    a = 0.46850 / (z1**0.23 + z2**0.23)
    phi = np.sum(ZBL_C * np.exp(-ZBL_E * r / a))
    return 14.4 * z1 * z2 / r * phi * 0.5 * (np.cos(np.pi * r / rc) + 1) if r < rc else 0.0


def exp_pair(r, z1, z2, rc):
    from ase.data import covalent_radii

    r1, r2 = 0.8 * covalent_radii[z1], 0.8 * covalent_radii[z2]
    v = 100.0**2 * np.exp(-r * (r1 + r2) / (r1 * r2)) / r**2
    return v * 0.5 * (np.cos(np.pi * r / rc) + 1) if r < rc else 0.0


def correction_energy(term, pos, Z):
    """Full neighbour list over all ordered pairs, plus one padding entry (0, 0) that must be masked."""
    n = len(Z)
    ij = np.array([(i, j) for i in range(n) for j in range(n) if i != j] + [(0, 0)]).T
    dr_vec = pos[ij[1]] - pos[ij[0]]
    args = (jnp.asarray(pos), jnp.asarray(dr_vec), jnp.asarray(Z), jnp.asarray(ij), jnp.zeros(3), {})
    params = term.init(jax.random.PRNGKey(0), *args)
    return float(term.apply(params, *args))


@pytest.mark.parametrize(
    "Term, pair, per_ordered_pair",
    [(ZBLRepulsion, zbl_pair, 0.5), (ExponentialRepulsion, exp_pair, 1.0)],
    ids=["zbl", "exponential"],
)
@pytest.mark.parametrize("Z", [(1, 1), (6, 8), (29, 79)])
def test_correction_dimer_closed_form(Term, pair, per_ordered_pair, Z):
    # ZBL halves the sum over ordered pairs; ExponentialRepulsion does not (each pair counted twice).
    rc = 1.5
    term = Term(r_max=rc, dtype="fp64")
    for r in [0.1, 0.4, 0.9, 1.3, 1.49, 1.5, 1.7, 3.0]:
        pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]) @ Rotation.random(random_state=int(10 * r)).as_matrix()
        E = correction_energy(term, pos, np.array(Z))
        ref = 2 * per_ordered_pair * pair(r, *Z, rc)
        if r >= rc:
            assert abs(E) < 1e-12
        else:
            np.testing.assert_allclose(E, ref, rtol=1e-12)


@pytest.mark.parametrize("Term, pair, per_ordered_pair", [(ZBLRepulsion, zbl_pair, 0.5), (ExponentialRepulsion, exp_pair, 1.0)])
def test_correction_is_pairwise_additive(Term, pair, per_ordered_pair):
    rc = 2.0
    atoms = random_cluster(6, species=(1, 6, 8), min_dist=0.6, box=2.5, seed=8)
    pos, Z = atoms.positions, atoms.numbers
    E = correction_energy(Term(r_max=rc, dtype="fp64"), pos, Z)
    ref = sum(
        2 * per_ordered_pair * pair(np.linalg.norm(pos[i] - pos[j]), Z[i], Z[j], rc)
        for i, j in itertools.combinations(range(len(Z)), 2)
    )
    assert ref > 0
    np.testing.assert_allclose(E, ref, rtol=1e-12)


# ---------------------------------------------------------------- property heads
def head_model(mode, aggregation="none"):
    head = dict(name="p", mode=mode, aggregation=aggregation, nn=[8], dtype="fp64")
    return model_config(property_heads=[head], calc_stress=False)


def rotated(atoms, seed):
    Q = Rotation.random(random_state=seed).as_matrix()
    b = atoms.copy()
    b.positions = atoms.positions @ Q.T
    if atoms.pbc.any():
        b.set_cell(atoms.cell.array @ Q.T, scale_atoms=False)
    return b, Q


@pytest.mark.parametrize(
    "atoms",
    [
        pytest.param(random_cluster(10, seed=9), id="cluster"),
        pytest.param(
            random_bulk("triclinic", rep=2, seed=10), id="periodic",
            marks=pytest.mark.xfail(strict=True, reason="l1/l2 property heads use fractional coordinates for "
                                    "periodic systems, so vector/tensor outputs do not rotate with the structure"),
        ),
    ],
)
@pytest.mark.parametrize("mode", ["l0", "l1", "symmetric_l2"])
def test_property_head_rotates_correctly(atoms, mode):
    cfg = head_model(mode)
    builder, params = build(cfg, atoms)
    b, Q = rotated(atoms, seed=11)
    p, p_rot = offset_path(builder, params, atoms)["p"], offset_path(builder, params, b)["p"]
    expected = {"l0": p, "l1": p @ Q.T, "symmetric_l2": Q @ p @ Q.T}[mode]
    assert np.abs(p).max() > 1e-3
    np.testing.assert_allclose(p_rot, expected, atol=1e-10 * np.abs(p).max())


def test_sum_aggregation_is_extensive():
    atoms = random_bulk("monoclinic", rep=2, seed=12)
    builder, params = build(head_model("l0", "sum"), atoms)
    p1 = offset_path(builder, params, atoms)["p"]
    p3 = offset_path(builder, params, atoms.repeat((3, 1, 1)))["p"]
    assert np.abs(p1).max() > 1e-3
    np.testing.assert_allclose(p3, 3 * p1, rtol=1e-10)
