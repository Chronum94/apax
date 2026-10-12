import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space

from apax.config.md_config import NPTOptions
from apax.md.sim_utils import SimulationFunctions, System
from apax.md.simulate import run_sim

R_CUT = 3.0


class RecordingTraj:
    """Stand-in for TrajHandler that checks every sampled neighbor list
    against brute force and fails instead of looping on reallocation."""

    def __init__(self, path):
        self.traj_path = path / "traj.h5"
        self.n_reallocations = 0
        self.max_missing_pairs = 0
        self.box = None

    def step(self, args):
        _, predictions, nbr_kwargs = args
        self.box = nbr_kwargs["box"]
        missing = int(predictions["missing_pairs"])
        self.max_missing_pairs = max(self.max_missing_pairs, missing)

    def reset_buffer(self):
        # samples from the discarded chunk do not count
        self.max_missing_pairs = 0
        self.n_reallocations += 1
        assert self.n_reallocations < 10, "neighbor list reallocation loop"

    def write(self):
        pass

    def close(self):
        pass


def test_npt_compression_reallocates_neighbors(tmp_path):
    n_side, spacing = 4, 2.2
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

    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=True)

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
        box,
        R_CUT,
        0.3,
        capacity_multiplier=1.0,
        fractional_coordinates=True,
        format=partition.Sparse,
    )
    sim_fns = SimulationFunctions(energy_fn, auxiliary_fn, shift_fn, neighbor_fn)
    ensemble = NPTOptions(
        name="npt",
        dt=1.0,
        temperature_schedule={"name": "constant", "T0": 300},
        pressure=3e5,
        barostat_chain={"tau": 50},
    )
    traj = RecordingTraj(tmp_path)

    run_sim(
        system,
        sim_fns,
        ensemble,
        tmp_path,
        n_steps=200,
        n_inner=20,
        extra_capacity=0,
        rng_key=jax.random.PRNGKey(0),
        traj_handler=traj,
        sampling_rate=5,
        restart=False,
        disable_pbar=True,
    )

    assert traj.n_reallocations >= 1
    assert traj.box[0, 0] < 0.9 * L
    assert traj.max_missing_pairs == 0
