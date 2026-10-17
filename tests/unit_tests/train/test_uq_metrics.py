import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apax.config.train_config import MetricsConfig
from apax.train.metrics import initialize_metrics

REDUCTIONS = ["nll", "rll", "rll_cal", "calib_alpha"]


def _forces_batch(rng, sigma_fn, n_struct=3, n_atoms=5, n_pad=2):
    """Per-component errors drawn from N(0, s^2) with a known per-atom s; the
    last `n_pad` atom slots are padding with garbage values."""
    s = rng.uniform(0.02, 0.3, size=(n_struct, n_atoms, 1)) * np.ones(3)
    err = rng.normal(size=s.shape) * s
    sigma = sigma_fn(err, s)
    numbers = np.ones((n_struct, n_atoms), dtype=int)
    numbers[:, n_atoms - n_pad :] = 0
    err[:, n_atoms - n_pad :] = 123.0
    sigma[:, n_atoms - n_pad :] = 0.0
    inputs = {"numbers": jnp.array(numbers)}
    label = {"forces": jnp.zeros(s.shape)}
    pred = {"forces": jnp.array(err), "forces_uncertainty": jnp.array(sigma)}
    real = numbers > 0
    return inputs, label, pred, err[real], sigma[real]


def _compute(inputs, label, pred):
    Metrics = initialize_metrics([MetricsConfig(name="forces", reductions=REDUCTIONS)])
    m = Metrics.single_from_model_output(inputs=inputs, label=label, prediction=pred)
    return {k: float(v) for k, v in m.compute().items()}


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def _numpy_scores(e, s):
    nll = lambda sig: np.mean(0.5 * np.log(2 * np.pi * sig**2) + 0.5 * e**2 / sig**2)
    base = nll(np.full_like(e, np.sqrt(np.mean(e**2))))
    oracle = nll(np.abs(e))
    alpha = np.sqrt(np.mean(e**2 / s**2))
    return {
        "forces_nll": nll(s),
        "forces_rll": 100 * (nll(s) - base) / (oracle - base),
        "forces_rll_cal": 100 * (nll(alpha * s) - base) / (oracle - base),
        "forces_calib_alpha": alpha,
    }


def test_scores_match_numpy_and_ignore_padding():
    rng = np.random.default_rng(0)
    inputs, label, pred, e, s = _forces_batch(rng, lambda err, s: 1.7 * s)
    got = _compute(inputs, label, pred)
    ref = _numpy_scores(e, s)
    for k, v in ref.items():
        np.testing.assert_allclose(got[k], v, rtol=1e-10)


def test_rll_limits():
    rng = np.random.default_rng(1)
    # constant sigma = RMSE: 0 %
    inputs, label, pred, e, _ = _forces_batch(rng, lambda err, s: np.ones_like(s))
    rmse = np.sqrt(np.mean(e**2))
    pred["forces_uncertainty"] = jnp.where(
        inputs["numbers"][..., None] > 0, rmse, 0.0
    ) * jnp.ones(3)
    np.testing.assert_allclose(_compute(inputs, label, pred)["forces_rll"], 0.0, atol=1e-9)
    # oracle sigma = |error|: 100 %
    inputs, label, pred, _, _ = _forces_batch(rng, lambda err, s: np.abs(err))
    np.testing.assert_allclose(
        _compute(inputs, label, pred)["forces_rll"], 100.0, atol=1e-9
    )


def test_calibration_recovers_scale():
    # sigma 3x too small everywhere: alpha = 3 (statistically), calibrated RLL equals
    # the RLL of the correct sigmas
    rng = np.random.default_rng(2)
    inputs, label, pred, e, s = _forces_batch(
        rng, lambda err, s: s / 3, n_struct=200, n_atoms=20
    )
    got = _compute(inputs, label, pred)
    assert abs(got["forces_calib_alpha"] - 3) < 0.1
    good = _numpy_scores(e, 3 * s)
    np.testing.assert_allclose(got["forces_rll_cal"], good["forces_rll_cal"], rtol=1e-10)
    assert got["forces_rll"] < got["forces_rll_cal"]


def test_merge_equals_single_batch():
    rng = np.random.default_rng(3)
    Metrics = initialize_metrics([MetricsConfig(name="forces", reductions=REDUCTIONS)])
    inputs, label, pred, e, s = _forces_batch(rng, lambda err, s: s)
    halves = []
    for sl in (slice(0, 1), slice(1, 3)):
        halves.append(
            Metrics.single_from_model_output(
                inputs={k: v[sl] for k, v in inputs.items()},
                label={k: v[sl] for k, v in label.items()},
                prediction={k: v[sl] for k, v in pred.items()},
            )
        )
    merged = halves[0].merge(halves[1]).compute()
    single = _numpy_scores(e, s)
    for k, v in single.items():
        np.testing.assert_allclose(float(merged[k]), v, rtol=1e-10)


def test_unavailable_uncertainty():
    cfg = [MetricsConfig(name="forces", reductions=["mae", "rll"])]
    with pytest.raises(ValueError, match="forces_uncertainty"):
        initialize_metrics(cfg, uncertainties={"energy"})
    Metrics = initialize_metrics(cfg, uncertainties={"energy"}, strict=False)
    assert list(Metrics.__annotations__) == ["forces_mae"]
