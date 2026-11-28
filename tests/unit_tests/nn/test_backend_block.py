"""L2 atom-blocked backend (EnergyModel.backend_block, md_config.backend_block_atoms).

CPU: the automatic block size (warp_contract.auto_backend_block).
GPU (+ warp): a GMNN with Warp moment and contraction kernels gives the same energy and
forces with the contraction + readout run over remat'ed atom blocks (uneven last block,
block >= n_atoms) as unblocked, to fp32 round-off."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk
from jax_md import partition

from apax.layers.descriptor.warp_contract import auto_backend_block

MIB = 2**20


# ------------------------------------------------------------------ CPU: block-size rule


def test_auto_block_matches_l40s_calibration():
    # 6 A benchmark model (K 12, C 5, n_contr 8, 128-128 readout) on a 96 MiB L2: the
    # measured optimum was ~16k-atom blocks at 69k / 104k atoms
    args = (12, 5, 8, (), (128, 128), 96 * MIB)
    b = auto_backend_block(104247, *args)
    assert 13000 <= b <= 17000
    n_blocks = -(-104247 // b)
    assert b * (n_blocks - 1) < 104247 <= b * n_blocks  # equal blocks: no small tail
    assert 104247 - b * (n_blocks - 1) > 0.9 * b


def test_auto_block_off_for_small_systems_or_unknown_l2():
    args = (12, 5, 8, (), (128, 128))
    assert auto_backend_block(30888, *args, 96 * MIB) == 0  # < 3 blocks: no gain
    assert auto_backend_block(10**6, *args, 0) == 0  # L2 not queryable
    # a larger cache or a smaller model -> larger blocks
    assert auto_backend_block(10**6, *args, 128 * MIB) > auto_backend_block(
        10**6, *args, 96 * MIB
    )
    assert auto_backend_block(10**6, 8, 4, 8, (), (64, 64), 96 * MIB) > (
        auto_backend_block(10**6, *args, 96 * MIB)
    )


# ------------------------------------------------------------------ GPU: blocked model

gpu = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="needs a CUDA GPU"
)

R_MAX = 5.0
MODEL = {
    "name": "gmnn",
    "basis": {"name": "bessel", "n_basis": 8, "r_max": R_MAX},
    "n_radial": 6,
    "n_radial_tensor": 3,
    "n_contr": 8,
    "nn": [32, 32],
    "packed_moments": True,
    "descriptor_dtype": "fp32",
    "readout_dtype": "fp32",
    "scale_shift_dtype": "fp64",
}


def _system():
    from apax.utils.space import periodic_general_elementwise

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((4, 4, 4))  # 512 atoms
    atoms.rattle(0.15, seed=3)
    box = np.asarray(atoms.cell.array.T)
    disp, _ = periodic_general_elementwise(box)
    disp.constant_box = True
    frac = jnp.asarray(atoms.get_scaled_positions(wrap=True), jnp.float32)
    nbr = partition.neighbor_list(
        disp, box, R_MAX, 0.5, fractional_coordinates=True, format=partition.Sparse
    ).allocate(frac, extra_capacity=8)
    return atoms, box, disp, frac, nbr.idx


@gpu
def test_blocked_backend_matches_unblocked():
    pytest.importorskip("warp")
    from apax.config.model_config import GMNNConfig
    from apax.layers.descriptor.radial_table import tabulate_radial

    atoms, box, disp, R, idx = _system()
    Z = jnp.asarray(atoms.numbers)
    N = len(atoms)
    offsets = jnp.zeros((idx.shape[1], 3))
    cfg = GMNNConfig(**MODEL)
    Builder = cfg.get_builder()
    kw = {
        "apply_mask": True,
        "init_box": box,
        "inference_disp_fn": disp,
        "geometry_dtype": jnp.float32,
    }
    base = cfg.model_dump()
    params = (
        Builder(base)
        .build_energy_model(**kw)
        .init(jax.random.PRNGKey(0), R, Z, idx, box, offsets)
    )
    exact = Builder({**base, "descriptor_dtype": "fp64"}).build_radial_function()
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        table = tabulate_radial(
            exact,
            jax.device_put(params["params"]["representation"]["radial_fn"], cpu),
            sorted(set(atoms.numbers.tolist())),
            dtype=jnp.float32,
        )

    def energy_and_forces(block):
        d = {
            **base,
            "moment_kernel": "warp",
            "contract_kernel": "warp",
            "neighbor_grad": "pull",
            "backend_block": block,
        }
        model = Builder(d, radial_fn=table).build_energy_model(**kw)

        def energy(r):
            return model.apply(params, r, Z, idx, box, offsets)[0]

        e, g = jax.jit(jax.value_and_grad(energy))(R)
        return float(e), np.asarray(g)

    e0, g0 = energy_and_forces(0)
    assert np.all(np.isfinite(g0)) and np.abs(g0).max() > 0
    # 200: 3 blocks with a short last one (112 atoms); N: one block, no remat
    for block in (200, N):
        e, g = energy_and_forces(block)
        assert abs(e - e0) < 1e-6 * max(1.0, abs(e0))
        np.testing.assert_allclose(g, g0, rtol=0, atol=1e-5 * np.abs(g0).max())


@gpu
def test_l2_query():
    from apax.layers.descriptor.warp_contract import l2_cache_bytes

    assert l2_cache_bytes(0) >= MIB  # every CUDA GPU in use has > 1 MiB of L2
