from types import SimpleNamespace

import pytest

from apax.config.model_config import ShallowEnsembleConfig
from apax.config.train_config import LossConfig, MetricsConfig
from apax.train.run import initialize_loss_fn, shallow_force_variance


def _config(losses, metrics=(), force_variance=None, n_members=8):
    ens = ShallowEnsembleConfig(n_members=n_members, force_variance=force_variance)
    cfg = SimpleNamespace(model=SimpleNamespace(ensemble=ens), metrics=list(metrics))
    return cfg, initialize_loss_fn([LossConfig(**l) for l in losses])


E_NLL = {"name": "energy", "loss_type": "nll"}
F_MSE = {"name": "forces", "loss_type": "mse"}
F_NLL = {"name": "forces", "loss_type": "nll"}
F_RLL = MetricsConfig(name="forces", reductions=["mae", "rll"])


@pytest.mark.parametrize(
    "losses, metrics, expected",
    [
        ([E_NLL, F_MSE], [], (False, False)),
        ([E_NLL, F_NLL], [], (True, True)),
        ([E_NLL, {"name": "forces", "loss_type": "nll_3x3"}], [], (True, True)),
        ([E_NLL, {"name": "forces", "loss_type": "crps"}], [], (True, True)),
        ([E_NLL, F_MSE], [F_RLL], (False, True)),
        ([E_NLL, F_MSE], [MetricsConfig(name="energy", reductions=["rll"])], (False, False)),
    ],
)
def test_inferred(losses, metrics, expected):
    cfg, loss_fn = _config(losses, metrics)
    assert shallow_force_variance(cfg, loss_fn) == expected


def test_explicit_override():
    cfg, loss_fn = _config([E_NLL, F_MSE], force_variance=True)
    assert shallow_force_variance(cfg, loss_fn) == (True, True)
    cfg, loss_fn = _config([E_NLL, F_NLL], force_variance=False)
    with pytest.raises(ValueError):
        shallow_force_variance(cfg, loss_fn)


def test_not_a_shallow_ensemble():
    cfg, loss_fn = _config([E_NLL, F_NLL], n_members=1)
    assert shallow_force_variance(cfg, loss_fn) == (None, None)
