import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import partition, space

from apax.config.md_config import LangevinOptions, NPTOptions, NVTOptions
from apax.md.replicas import allocate_neighbors, make_neighbor_update
from apax.md.sim_utils import SimulationFunctions, System
from apax.md.simulate import run_sim

R_CUT, SKIN = 3.0, 0.3


class RecordingTraj:
    """Keeps the sampled positions and counts neighbor pairs missing from the list."""

    def __init__(self, path):
        self.traj_path = path / "traj.h5"
        self.positions = []
        self.max_missing_pairs = 0
        self.n_resets = 0

    def step(self, args):
        state, predictions, _ = args
        self.positions.append(np.asarray(state.position))
        self.max_missing_pairs = max(
            self.max_missing_pairs, int(predictions["missing_pairs"])
        )

    def reset_buffer(self):
        self.positions = []
        self.max_missing_pairs = 0
        self.n_resets += 1
        assert self.n_resets < 10, "neighbor list reallocation loop"

    def write(self):
        pass

    def close(self):
        pass


def make_system(n_side=4, spacing=2.2):
    L = n_side * spacing
    box = jnp.eye(3) * L
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(*[np.arange(n_side)] * 3, indexing="ij"), -1)
    cart = (grid.reshape(-1, 3) + 0.5) * spacing
    cart += rng.normal(scale=0.05, size=cart.shape)
    n_atoms = len(cart)
    system = System(
        atomic_numbers=jnp.full(n_atoms, 18, dtype=jnp.int32),
        masses=jnp.full(n_atoms, 39.9),
        positions=space.transform(jnp.linalg.inv(box), jnp.asarray(cart)),
        box=box,
        momenta=None,
    )
    return system


def make_sim_fns(system, dr_threshold=SKIN):
    n_atoms = len(system.positions)
    displacement_fn, shift_fn = space.periodic_general(
        system.box, fractional_coordinates=True
    )

    def pair_vectors(R, idx, box, perturbation=None):
        disp = jax.vmap(
            lambda a, b: displacement_fn(a, b, box=box, perturbation=perturbation)
        )
        return disp(R[idx[1]], R[idx[0]])

    def energy_fn(R, neighbor, box, perturbation=None, **kwargs):
        idx = neighbor.idx
        r = jnp.linalg.norm(pair_vectors(R, idx, box, perturbation) + 1e-9, axis=-1)
        mask = (idx[0] < n_atoms) & (r < R_CUT)
        return 0.25 * jnp.sum(jnp.where(mask, (1 - r / R_CUT) ** 2, 0.0))

    def auxiliary_fn(R, Z, neighbor, box, offsets):
        idx = neighbor.idx
        r = jnp.linalg.norm(pair_vectors(R, idx, box) + offsets, axis=-1)
        n_listed = jnp.sum((idx[0] < n_atoms) & (r < R_CUT))
        all_pairs = space.map_product(lambda a, b: displacement_fn(a, b, box=box))
        r_all = jnp.linalg.norm(all_pairs(R, R), axis=-1) + 1e6 * jnp.eye(n_atoms)
        return {"missing_pairs": jnp.sum(r_all < R_CUT) - n_listed}

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        system.box,
        R_CUT,
        SKIN,
        capacity_multiplier=1.0,
        fractional_coordinates=True,
        format=partition.Sparse,
    )
    return SimulationFunctions(
        energy_fn, auxiliary_fn, shift_fn, neighbor_fn, R_CUT, dr_threshold
    )


def run(tmp_path, ensemble, n_replicas, rng_key, n_steps=100, dr_threshold=SKIN):
    system = make_system()
    trajs = []
    for k in range(n_replicas):
        path = tmp_path / f"replica_{k}"
        path.mkdir(parents=True)
        trajs.append(RecordingTraj(path))
    run_sim(
        system,
        make_sim_fns(system, dr_threshold),
        ensemble,
        tmp_path,
        n_steps=n_steps,
        n_inner=20,
        extra_capacity=0,
        rng_key=rng_key,
        traj_handler=trajs if n_replicas > 1 else trajs[0],
        sampling_rate=5,
        restart=False,
        disable_pbar=True,
        n_replicas=n_replicas,
    )
    return trajs


NVT = NVTOptions(name="nvt", dt=1.0, temperature_schedule={"name": "constant", "T0": 300})
LANGEVIN = LangevinOptions(
    name="langevin",
    dt=1.0,
    friction=0.01,
    temperature_schedule={"name": "constant", "T0": 300},
)


@pytest.mark.parametrize("ensemble", [NVT, LANGEVIN], ids=["nvt", "langevin"])
def test_replica_matches_single_run(tmp_path, ensemble):
    key = jax.random.PRNGKey(3)
    batched = run(tmp_path / "batched", ensemble, 3, key)

    # replicas start alike but get different momenta
    final = [t.positions[-1] for t in batched]
    assert not np.allclose(final[0], final[1], atol=1e-3)

    for k, replica_key in enumerate(jax.random.split(key, 3)):
        single = run(tmp_path / f"single_{k}", ensemble, 1, replica_key)[0]
        assert len(single.positions) == len(batched[k].positions)
        np.testing.assert_allclose(
            np.stack(batched[k].positions), np.stack(single.positions), atol=1e-4
        )
        assert batched[k].max_missing_pairs == 0


def test_replicas_without_skin_info_match(tmp_path):
    # without dr_threshold every step rebuilds through the vmapped select
    key = jax.random.PRNGKey(5)
    fast = run(tmp_path / "fast", NVT, 2, key)
    slow = run(tmp_path / "slow", NVT, 2, key, dr_threshold=None)
    for a, b in zip(fast, slow):
        np.testing.assert_allclose(
            np.stack(a.positions), np.stack(b.positions), atol=1e-4
        )


def test_npt_replicas_reallocate_like_single_runs(tmp_path):
    ensemble = NPTOptions(
        name="npt",
        dt=1.0,
        temperature_schedule={"name": "constant", "T0": 300},
        pressure=3e5,
        barostat_chain={"tau": 50},
    )
    key = jax.random.PRNGKey(0)
    batched = run(tmp_path / "batched", ensemble, 2, key, n_steps=200)
    for k, replica_key in enumerate(jax.random.split(key, 2)):
        single = run(tmp_path / f"single_{k}", ensemble, 1, replica_key, n_steps=200)[0]
        assert batched[k].n_resets >= 1
        np.testing.assert_allclose(
            np.stack(batched[k].positions), np.stack(single.positions), atol=1e-4
        )
        # jax_md's skin check ignores the box deformation, so a fast compression
        # can miss pairs (replica 1 does); replicas must not add to that
        assert batched[k].max_missing_pairs == single.max_missing_pairs


def test_allocate_neighbors_fits_every_replica():
    system = make_system()
    sim_fns = make_sim_fns(system)
    rng = np.random.default_rng(1)
    n_atoms = len(system.positions)
    # one dense and one loose replica need different capacities
    dense = np.asarray(system.positions) * 0.8
    loose = rng.uniform(size=(n_atoms, 3))
    positions = jnp.asarray(np.stack([loose, dense]))

    neighbors = allocate_neighbors(sim_fns.neighbor_fn, positions, {}, 0, True)
    assert not np.any(neighbors.did_buffer_overflow)
    for k in range(2):
        single = sim_fns.neighbor_fn.allocate(positions[k])
        idx = np.asarray(neighbors.idx[k])
        pairs = {tuple(p) for p in idx.T if p[0] < n_atoms}
        expected = {tuple(p) for p in np.asarray(single.idx).T if p[0] < n_atoms}
        assert pairs == expected


def test_neighbor_update_skips_unmoved_replicas():
    system = make_system()
    sim_fns = make_sim_fns(system)
    positions = jnp.stack([system.positions] * 2)
    neighbors = allocate_neighbors(sim_fns.neighbor_fn, positions, {}, 0, True)
    update = jax.jit(make_neighbor_update(2, True, SKIN, R_CUT))
    boxes = jnp.stack([system.box] * 2)

    # replica 1 moves one atom past half the skin; replica 0 stays put
    moved = positions.at[1, 0].add(jnp.array([0.2, 0.0, 0.0]) / system.box[0, 0])
    updated = update(neighbors, moved, {}, boxes)
    np.testing.assert_array_equal(updated.reference_position[0], positions[0])
    np.testing.assert_array_equal(updated.reference_position[1], moved[1])

    # nothing moved far enough: no rebuild at all
    nudged = positions + 0.01 / system.box[0, 0]
    updated = update(neighbors, nudged, {}, boxes)
    np.testing.assert_array_equal(updated.reference_position, positions)


def test_replicas_shard_over_devices(tmp_path):
    script = textwrap.dedent(
        f"""
        import jax, numpy as np
        from pathlib import Path
        assert jax.device_count() == 2
        import apax.md.replicas as replicas
        from tests.unit_tests.md import test_replicas as t

        placements = []
        shard = replicas.shard_replicas
        def recording_shard(tree, n):
            out = shard(tree, n)
            placements.append(jax.tree.leaves(out)[0].sharding)
            return out
        replicas.shard_replicas = recording_shard
        import apax.md.simulate as simulate
        simulate.shard_replicas = recording_shard

        path = Path({str(tmp_path)!r})
        trajs = t.run(path / "sharded", t.NVT, 2, jax.random.PRNGKey(3))
        assert len(placements[0].device_set) == 2, placements
        single = t.run(path / "single", t.NVT, 1, jax.random.split(jax.random.PRNGKey(3), 2)[1])[0]
        np.testing.assert_allclose(
            np.stack(trajs[1].positions), np.stack(single.positions), atol=1e-4
        )
        print("OK")
        """
    )
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2")
    env["JAX_PLATFORMS"] = "cpu"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "OK" in result.stdout
