"""MD step loop (run_sim): peeling the sampling step and unrolling the fori_loop
(loop_unroll) must not change the trajectory or which steps are sampled.

Reference: n_inner 40, sampling_rate 20 (sampling_rate % n_inner != 0: every step keeps
the sampling cond, nothing peeled). Peeled: n_inner 20, sampling_rate 20, unrolled 1, 2
and 3 times (3 does not divide the 19 non-sampling steps: remainder iterations)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import partition, space

from apax.config.md_config import NVTOptions
from apax.md.sim_utils import SimulationFunctions, System
from apax.md.simulate import run_sim

R_CUT = 3.0


class RecordingTraj:
    def __init__(self, path):
        self.traj_path = path / "traj.h5"
        self.positions = []

    def step(self, args):
        state, _, _ = args
        self.positions.append(np.asarray(state.position))

    def reset_buffer(self):
        raise AssertionError("no neighbour-list reallocation expected")

    def write(self):
        pass

    def close(self):
        pass


def _run(tmp_path, n_inner, sampling_rate, loop_unroll):
    n_side, spacing = 4, 2.2
    L = n_side * spacing
    box = jnp.eye(3) * L
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(*[np.arange(n_side)] * 3, indexing="ij"), -1)
    cart = (grid.reshape(-1, 3) + 0.5) * spacing + rng.normal(
        scale=0.05, size=(n_side**3, 3)
    )
    n_atoms = len(cart)
    system = System(
        atomic_numbers=jnp.full(n_atoms, 18, dtype=jnp.int32),
        masses=jnp.full(n_atoms, 39.9),
        positions=space.transform(jnp.linalg.inv(box), jnp.asarray(cart)),
        box=box,
        momenta=None,
    )
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=True)

    def energy_fn(R, neighbor, box, perturbation=None, **kwargs):
        idx = neighbor.idx
        d = jax.vmap(lambda a, b: displacement_fn(a, b, box=box))(R[idx[1]], R[idx[0]])
        r = jnp.linalg.norm(d + 1e-9, axis=-1)
        mask = (idx[0] < n_atoms) & (r < R_CUT)
        return 0.25 * jnp.sum(jnp.where(mask, (1 - r / R_CUT) ** 2, 0.0))

    def auxiliary_fn(R, Z, neighbor, box, offsets):
        return {"energy": energy_fn(R, neighbor, box)}

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box,
        R_CUT,
        0.3,
        capacity_multiplier=1.5,
        fractional_coordinates=True,
        format=partition.Sparse,
        disable_cell_list=True,  # 64 atoms
    )
    sim_fns = SimulationFunctions(energy_fn, auxiliary_fn, shift_fn, neighbor_fn)
    ensemble = NVTOptions(
        name="nvt", dt=1.0, temperature_schedule={"name": "constant", "T0": 300}
    )
    traj = RecordingTraj(tmp_path)
    run_sim(
        system,
        sim_fns,
        ensemble,
        tmp_path,
        n_steps=200,
        n_inner=n_inner,
        extra_capacity=0,
        rng_key=jax.random.PRNGKey(0),
        traj_handler=traj,
        sampling_rate=sampling_rate,
        loop_unroll=loop_unroll,
        restart=False,
        disable_pbar=True,
    )
    return np.stack(traj.positions)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return _run(
        tmp_path_factory.mktemp("ref"), n_inner=40, sampling_rate=20, loop_unroll=1
    )


@pytest.mark.parametrize("loop_unroll", [1, 2, 3])
def test_peeled_unrolled_loop_matches_reference(tmp_path, reference, loop_unroll):
    peeled = _run(tmp_path, n_inner=20, sampling_rate=20, loop_unroll=loop_unroll)
    assert peeled.shape == reference.shape  # same steps sampled (0, 20, ..., 180)
    assert reference.shape[0] == 10
    # identical maths; only XLA's fusion of the unrolled bodies may reorder round-off
    np.testing.assert_allclose(peeled, reference, rtol=0, atol=1e-6)
    assert np.abs(reference[-1] - reference[0]).max() > 1e-3  # atoms actually moved


def test_unrolled_unpeeled_loop_matches_reference(tmp_path, reference):
    # sampling_rate % n_inner != 0 keeps the sampling cond in the loop; unroll still applies
    out = _run(tmp_path, n_inner=40, sampling_rate=20, loop_unroll=2)
    np.testing.assert_allclose(out, reference, rtol=0, atol=1e-6)
