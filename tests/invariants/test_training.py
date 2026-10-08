"""Training-pipeline invariants: losses/metrics vs numpy, shift fits, data splits, checkpoints,
transfer learning, resume. Tiny EMT datasets, few epochs."""

import csv
import shutil
from importlib import resources
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import yaml
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from flax.traverse_util import flatten_dict

from apax.cli import templates
from apax.config import parse_config
from apax.data.statistics import IsolatedAtomEnergyShift, PerElementRegressionShift
from apax.train.checkpoints import load_params, restore_parameters
from apax.train.eval import load_test_data
from apax.train.loss import Loss
from apax.train.metrics import initialize_metrics
from apax.train.run import run
from apax.utils.data import split_idxs
from apax.utils.random import seed_py_np_tf

N_DATA, N_TRAIN, N_VALID = 12, 6, 3


# ---------------------------------------------------------------- helpers (conftest candidates)
def emt_dataset(n=N_DATA):
    """Cu/Au fcc cells of 4 and 8 atoms (so batches are padded), rattled, EMT-labelled."""
    rng, out = np.random.default_rng(0), []
    for i in range(n):
        a = bulk("Cu", "fcc", a=3.6, cubic=True)
        a.numbers[rng.integers(0, 4)] = 79
        if i % 2:
            a = a.repeat((2, 1, 1))
        a.rattle(0.1, seed=i)
        a.calc = EMT()
        a.calc = SinglePointCalculator(a, energy=a.get_potential_energy(), forces=a.get_forces())
        out.append(a)
    return out


def train_config(root, experiment, **top):
    cfg = dict(
        n_epochs=2, seed=1, ckpt_interval=1,
        data=dict(directory=str(root), experiment=experiment, data_path=str(root / "ds.extxyz"),
                  n_train=N_TRAIN, n_valid=N_VALID, batch_size=2, valid_batch_size=3),
        model=dict(name="gmnn", nn=[8, 8], n_radial=3, basis=dict(name="bessel", n_basis=4, r_max=4.0),
                   descriptor_dtype="fp64", readout_dtype="fp64"),
        loss=[dict(name="energy"), dict(name="forces")],
        metrics=[dict(name="energy", reductions=["mae"]), dict(name="forces", reductions=["mae"])],
        progress_bar=dict(disable_epoch_pbar=True),
    )
    cfg.update(top)
    return cfg


def train(cfg):
    with jax.enable_x64(True):
        run(cfg)
    return parse_config(cfg).data.model_version_path


def flat(params):
    return {k: np.asarray(v) for k, v in flatten_dict(params, sep="/").items()}


def read_log(path):
    with open(path / "log.csv") as f:
        return [{k: v for k, v in r.items() if k != "epoch_time"} for r in csv.DictReader(f)]


ZERO_LR = dict(emb_lr=0.0, nn_lr=0.0, scale_lr=0.0, shift_lr=0.0)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    write(root / "ds.extxyz", emt_dataset())
    return root


@pytest.fixture(scope="module")
def base(root):
    """3 epochs, checkpoint every epoch. Shared by checkpoint, split, transfer and resume tests."""
    return train(train_config(root, "base", n_epochs=3))


# ---------------------------------------------------------------- losses vs numpy
B, A = 3, 5
N_ATOMS = np.array([5, 3, 2])
MASK = np.arange(A)[None, :] < N_ATOMS[:, None]  # (B, A) real atoms


def padded_forces(rng):
    return rng.normal(size=(B, A, 3)) * MASK[..., None]


def batch(seed=0):
    rng = np.random.default_rng(seed)
    inputs = {"n_atoms": N_ATOMS, "numbers": np.where(MASK, 29, 0)}
    label = {"energy": rng.normal(size=B), "forces": padded_forces(rng)}
    pred = {"energy": rng.normal(size=B), "forces": padded_forces(rng)}
    return inputs, label, pred


def cos_per_atom(f0, f1):
    """1 - cos angle for real atoms only, list of 1d arrays."""
    return [1 - np.sum(f0[b, :n] * f1[b, :n], -1)
            / np.linalg.norm(f0[b, :n], axis=-1) / np.linalg.norm(f1[b, :n], axis=-1)
            for b, n in enumerate(N_ATOMS)]


def ref_loss(name, kind, label, pred, exponent, delta=0.3):
    d0, d1 = label[name], pred[name]
    if name == "energy":
        per = {"mse": (d0 - d1) ** 2}
        if kind == "huber":
            x = np.abs(d0 - d1)
            per["huber"] = np.where(x > delta, delta * (x - 0.5 * delta), 0.5 * x**2)
        return np.mean(per[kind] / N_ATOMS**exponent)
    norms = [np.linalg.norm(d0[b, :n], axis=-1) for b, n in enumerate(N_ATOMS)]
    if kind == "mse":
        per = [np.sum((d0[b, :n] - d1[b, :n]) ** 2) for b, n in enumerate(N_ATOMS)]
    elif kind == "huber":
        x = [np.abs(d0[b, :n] - d1[b, :n]) for b, n in enumerate(N_ATOMS)]
        per = [np.sum(np.where(v > delta, delta * (v - 0.5 * delta), 0.5 * v**2)) for v in x]
    else:
        c = cos_per_atom(d0, d1)
        w = {"cosine_sim": [1.0] * B,
             "cosine_sim_div_magnitude": [1 / n for n in norms],
             "cosine_sim_exp_magnitude": [np.exp(-n) for n in norms]}[kind]
        per = [np.sum(ci * wi) for ci, wi in zip(c, w)]
    return np.mean(np.array(per) / N_ATOMS**exponent)


PADDED_COSINE = pytest.mark.xfail(
    strict=True, reason="padded atoms add 1 each to cosine_sim / cosine_sim_exp_magnitude losses")


@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("name,kind", [
    ("energy", "mse"), ("energy", "huber"), ("forces", "mse"), ("forces", "huber"),
    ("forces", "cosine_sim_div_magnitude"),
    pytest.param("forces", "cosine_sim", marks=PADDED_COSINE),
    pytest.param("forces", "cosine_sim_exp_magnitude", marks=PADDED_COSINE),
])
def test_loss_matches_numpy_on_padded_batch(name, kind, exponent):
    inputs, label, pred = batch()
    loss = Loss(name, kind, weight=1.7, atoms_exponent=exponent, parameters={"delta": 0.3})
    expected = 1.7 * ref_loss(name, kind, label, pred, exponent)
    np.testing.assert_allclose(loss(inputs, pred, label), expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["cosine_sim", "cosine_sim_exp_magnitude", "cosine_sim_div_magnitude"])
def test_cosine_losses_match_numpy_unpadded(kind):
    """Passing variant of the padded cosine xfail: every row full."""
    global N_ATOMS, MASK
    saved = N_ATOMS, MASK
    N_ATOMS, MASK = np.full(B, A), np.ones((B, A), bool)
    try:
        inputs, label, pred = batch(1)
        expected = ref_loss("forces", kind, label, pred, 1.0)
        np.testing.assert_allclose(Loss("forces", kind)(inputs, pred, label), expected, rtol=1e-12)
    finally:
        N_ATOMS, MASK = saved


# ---------------------------------------------------------------- metrics vs numpy
def metrics(name, reductions, inputs, label, pred):
    M = initialize_metrics([SimpleNamespace(name=name, reductions=reductions)])
    return M.single_from_model_output(inputs=inputs, label=label, prediction=pred).compute()


def test_energy_metrics_match_numpy():
    inputs, label, pred = batch()
    dE = label["energy"] - pred["energy"]
    out = metrics("energy", ["mae", "rmse", "per_atom_mae"], inputs, label, pred)
    np.testing.assert_allclose(out["energy_mae"], np.mean(np.abs(dE)), rtol=1e-12)
    np.testing.assert_allclose(out["energy_rmse"], np.sqrt(np.mean(dE**2)), rtol=1e-12)
    np.testing.assert_allclose(out["energy_per_atom_mae"], np.sum(np.abs(dE)) / N_ATOMS.sum(), rtol=1e-12)


@pytest.mark.xfail(strict=True, reason="per_atom_rmse is sqrt(sum dE^2 / sum N) (eV/sqrt(atom)), not an RMSE of dE/N")
def test_per_atom_rmse_is_rmse_over_n_for_equal_sizes():
    rng = np.random.default_rng(3)
    inputs = {"n_atoms": np.full(4, 8)}
    label, pred = {"energy": rng.normal(size=4)}, {"energy": rng.normal(size=4)}
    out = metrics("energy", ["rmse", "per_atom_rmse"], inputs, label, pred)
    np.testing.assert_allclose(out["energy_per_atom_rmse"], out["energy_rmse"] / 8, rtol=1e-12)


def test_metrics_merge_equal_batches_is_global_mean():
    inputs, label, pred = batch()
    M = initialize_metrics([SimpleNamespace(name="energy", reductions=["mae", "rmse"])])
    acc = M.empty()
    for b in range(B):  # one structure per batch
        s = lambda d: {k: v[b:b + 1] for k, v in d.items()}  # noqa: E731
        acc = acc.merge(M.single_from_model_output(inputs=s(inputs), label=s(label), prediction=s(pred)))
    dE = label["energy"] - pred["energy"]
    out = acc.compute()
    np.testing.assert_allclose(out["energy_mae"], np.mean(np.abs(dE)), rtol=1e-12)
    np.testing.assert_allclose(out["energy_rmse"], np.sqrt(np.mean(dE**2)), rtol=1e-12)


def force_metric_refs(label, pred):
    dF = (label["forces"] - pred["forces"])[MASK]
    c = np.concatenate(cos_per_atom(label["forces"], pred["forces"]))
    return {"forces_mae": np.mean(np.abs(dF)), "forces_rmse": np.sqrt(np.mean(dF**2)),
            "forces_cosine_sim": np.mean(c)}


PADDED_METRIC = pytest.mark.xfail(
    strict=True, reason="force metrics average over padded atoms (MAE/RMSE diluted, cosine_sim inflated)")


@pytest.mark.parametrize("padded", [False, pytest.param(True, marks=PADDED_METRIC)])
def test_force_metrics_match_numpy_over_real_atoms(padded):
    global N_ATOMS, MASK
    saved = N_ATOMS, MASK
    if not padded:
        N_ATOMS, MASK = np.full(B, A), np.ones((B, A), bool)
    try:
        inputs, label, pred = batch(2)
        out = metrics("forces", ["mae", "rmse", "cosine_sim"], inputs, label, pred)
        for k, v in force_metric_refs(label, pred).items():
            np.testing.assert_allclose(out[k], v, rtol=1e-12, err_msg=k)
    finally:
        N_ATOMS, MASK = saved


# ---------------------------------------------------------------- shifts
def test_per_element_regression_recovers_exact_atomic_energies():
    rng = np.random.default_rng(0)
    e_Z = {1: -13.6, 6: -1030.2, 8: -2041.7, 29: -45000.3}
    numbers = [rng.choice(list(e_Z), size=rng.integers(3, 12)) for _ in range(40)]
    energies = np.array([sum(e_Z[z] for z in Z) for Z in numbers])
    inputs = {"numbers": numbers, "n_atoms": np.array([len(Z) for Z in numbers])}
    shift = PerElementRegressionShift.compute(inputs, {"energy": energies}, {"energy_regularisation": 1e-12})
    np.testing.assert_allclose(shift[list(e_Z)], list(e_Z.values()), rtol=1e-10)


def test_isolated_atom_energy_shift_sets_E0s():
    E0s = {1: -0.5, 8: -75.1, 79: -3.3}
    shift = IsolatedAtomEnergyShift.compute(None, None, {"E0s": E0s})
    expected = np.zeros(119)
    expected[list(E0s)] = list(E0s.values())
    np.testing.assert_array_equal(shift, expected)


# ---------------------------------------------------------------- data splits
def test_split_reproducible_disjoint_and_within_data(base):
    idxs = np.load(base / "train_val_idxs.npz")
    tr, va = idxs["train_idxs"], idxs["val_idxs"]
    assert len(tr) == N_TRAIN and len(va) == N_VALID
    assert len(np.unique(np.concatenate([tr, va]))) == N_TRAIN + N_VALID
    assert set(tr) | set(va) <= set(range(N_DATA))
    seed_py_np_tf(1)  # config seed
    tr2, va2 = split_idxs(range(N_DATA), N_TRAIN, N_VALID)
    np.testing.assert_array_equal(tr, tr2)
    np.testing.assert_array_equal(va, va2)


@pytest.mark.xfail(strict=True, reason="apax eval's test set includes the validation structures")
def test_eval_test_set_disjoint_from_train_and_val(base, tmp_path):
    config = parse_config(base / "config.yaml")
    load_test_data(config, base, tmp_path)
    test = set(np.load(tmp_path / "test_idxs.npz")["test_idxs"])
    idxs = np.load(base / "train_val_idxs.npz")
    assert test == set(range(N_DATA)) - set(idxs["train_idxs"]) - set(idxs["val_idxs"])


@pytest.mark.xfail(strict=True, reason="apax eval with test_data_path drops the last structure (n_test=-1 slices [:-1])")
def test_eval_keeps_all_structures_of_test_data_path(root, tmp_path):
    cfg = train_config(root, "x")
    cfg["data"]["test_data_path"] = str(root / "ds.extxyz")
    assert len(load_test_data(parse_config(cfg), tmp_path, tmp_path / "eval")) == N_DATA


# ---------------------------------------------------------------- checkpoints
def test_best_checkpoint_is_min_val_loss_and_reproduces_logged_metric(base):
    """Reload best/ through the public ASE calculator; its val energy MAE matches the log at the best epoch."""
    from apax.md.ase_calc import ASECalculator
    from apax.utils.data import load_data

    log = read_log(base)
    best = min(log, key=lambda r: float(r["val_loss"]))
    with jax.enable_x64(True):
        calc = ASECalculator(base)
        val = [load_data(base.parent / "ds.extxyz")[i] for i in np.load(base / "train_val_idxs.npz")["val_idxs"]]
        err = [a.get_potential_energy() - calc.get_potential_energy(a.copy()) for a in val]
    np.testing.assert_allclose(np.mean(np.abs(err)), float(best["val_energy_mae"]), rtol=2e-5, atol=1e-5)


# ---------------------------------------------------------------- transfer learning
def tl_config(root, base, experiment, optimizer=None, **tl):
    return train_config(root, experiment, n_epochs=1, optimizer=optimizer or {},
                        transfer_learning=dict(base_model_checkpoint=str(base), **tl))


def test_transfer_with_zero_lr_reproduces_base_bitwise(root, base):
    ft = train(tl_config(root, base, "ft_zero", optimizer=ZERO_LR))
    p0, p1 = flat(load_params(base)), flat(load_params(ft))
    for k in p0:
        np.testing.assert_array_equal(p0[k], p1[k], err_msg=k)


def test_freeze_keeps_params_and_reset_reinitialises(root, base):
    ft = train(tl_config(root, base, "ft_freeze", freeze_layers=["dense_0"], reset_layers=["dense_2"]))
    p0, p1 = flat(load_params(base)), flat(load_params(ft))
    frozen = [k for k in p0 if "dense_0" in k]
    assert frozen
    for k in p0:
        if k in frozen:
            np.testing.assert_array_equal(p0[k], p1[k], err_msg=k)
        elif "scale_shift" not in k:  # trained (embedding, dense_1) or reset+trained (dense_2)
            assert not np.array_equal(p0[k], p1[k]), k


def test_reset_layer_equals_fresh_init(root, base):
    """reset + frozen layer must equal the layer of a fresh run with the same seed at epoch 0."""
    ft = train(tl_config(root, base, "ft_reset", freeze_layers=["dense_1"], reset_layers=["dense_1"]))
    fresh = train(train_config(root, "fresh", n_epochs=1, optimizer=ZERO_LR))
    p_ft, p_fresh, p_base = flat(load_params(ft)), flat(load_params(fresh)), flat(load_params(base))
    k = "params/energy_model/readout/dense_1/w"
    np.testing.assert_array_equal(p_ft[k], p_fresh[k])
    assert not np.array_equal(p_ft[k], p_base[k])


# ---------------------------------------------------------------- resume
@pytest.mark.xfail(strict=True, reason="resuming from latest/ is not deterministic (params/log differ from an uninterrupted run)")
def test_resume_matches_uninterrupted_run(root, base):
    """Simulate a crash after epoch 1: drop latest/2, resume to n_epochs=3, compare with the full run."""
    resumed = root / "resumed"
    shutil.copytree(base, resumed)
    shutil.rmtree(resumed / "latest" / "2")
    log_full = read_log(base)
    train(train_config(root, "resumed", n_epochs=3))
    p0 = flat(load_params(base / "latest", best=False))
    p1 = flat(load_params(resumed / "latest", best=False))
    for k in p0:
        np.testing.assert_array_equal(p0[k], p1[k], err_msg=k)
    assert read_log(resumed) == log_full


# ---------------------------------------------------------------- template
@pytest.mark.xfail(strict=True, reason="`apax template train --full` config crashes in training")
def test_full_template_trains_one_epoch(root, tmp_path):
    cfg = yaml.safe_load(resources.files(templates).joinpath("train_config_full.yaml").read_text())
    cfg["n_epochs"] = 1
    cfg["data"].update(directory=str(tmp_path), data_path=str(root / "ds.extxyz"),
                       n_train=N_TRAIN, n_valid=N_VALID, valid_batch_size=3)
    cfg["progress_bar"]["disable_epoch_pbar"] = True
    path = train(cfg)
    assert (path / "best").is_dir()
