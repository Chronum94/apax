import logging
from functools import partial

import flax
import jax.numpy as jnp
import numpy as np
from clu import metrics

from apax.utils.math import normed_dotp

log = logging.getLogger(__name__)


class Averagefp64(metrics.Average):
    @classmethod
    def empty(cls) -> metrics.Metric:
        return cls(total=jnp.array(0, jnp.float64), count=jnp.array(0, jnp.int64))


class RootAverage(Averagefp64):
    """
    Modifies the `compute` method of `metrics.Average` to obtain the root of the average.
    Meant to be used with `mse_fn`.
    """

    def compute(self) -> jnp.array:
        return jnp.sqrt(self.total / self.count)


def _mask(inputs: dict, values: jnp.array) -> jnp.array:
    """
    Mask of real (non-padding) entries along the leading two axes of `values`.
    Per-atom arrays (batch, max_atoms, ...) are masked by padded atom slots,
    structure-level arrays (batch, ...) are left unmasked.
    """
    numbers = inputs["numbers"]
    if values.ndim >= 2 and values.shape[:2] == numbers.shape:
        return numbers > 0
    return jnp.ones(values.shape[:1], dtype=bool)


def mae_fn(
    inputs: dict, label: dict[jnp.array], prediction: dict[jnp.array], key: str
) -> dict:
    """
    Computes the Mean Absolute Error of two arrays, ignoring padded atoms.
    """
    values = jnp.abs(label[key] - prediction[key])
    return dict(values=values, mask=_mask(inputs, values))


def mse_fn(
    inputs: dict, label: dict[jnp.array], prediction: dict[jnp.array], key: str
) -> dict:
    """
    Computes the Mean Squared Error of two arrays, ignoring padded atoms.
    """
    values = (label[key] - prediction[key]) ** 2
    return dict(values=values, mask=_mask(inputs, values))


def cosine_sim(
    inputs: dict, label: dict[jnp.array], prediction: dict[jnp.array], key: str
) -> dict:
    """
    Computes the cosine similarity of two arrays, ignoring padded atoms.
    """
    values = 1.0 - normed_dotp(label[key], prediction[key])
    return dict(values=values, mask=_mask(inputs, values))


def per_atom_mae_fn(
    inputs: dict, label: dict[jnp.array], prediction: dict[jnp.array], key: str
) -> dict:
    """
    Computes the per atom Mean Absolute Error of two arrays.
    Only reasanable when using with structural
    properties like 'energy'.
    """
    values = jnp.abs(label[key] - prediction[key]) / inputs["n_atoms"]
    return dict(values=values, mask=_mask(inputs, values))


def per_atom_mse_fn(
    inputs: dict, label: dict[jnp.array], prediction: dict[jnp.array], key: str
) -> dict:
    """
    Computes the per atom Mean Squared Error of two arrays.
    Only reasanable when using with structural
    properties like 'energy'.
    """
    values = ((label[key] - prediction[key]) / inputs["n_atoms"]) ** 2
    return dict(values=values, mask=_mask(inputs, values))


@flax.struct.dataclass
class GaussianScores(metrics.Metric):
    """Sums needed to score Gaussian uncertainties: the NLL of the model, of a constant
    sigma equal to the RMSE (baseline) and of sigma = |error| (oracle), and the
    closed-form post-hoc calibration factor alpha^2 = mean(error^2 / sigma^2).
    Constant 0.5 log(2 pi) terms are left out; they cancel in the RLL.
    """

    log_var: jnp.ndarray
    z2: jnp.ndarray
    err2: jnp.ndarray
    log_err2: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def empty(cls):
        zero = jnp.array(0.0, jnp.float64)
        return cls(zero, zero, zero, zero, jnp.array(0, jnp.int64))

    @classmethod
    def from_model_output(cls, err2, var, mask=None, **_):
        err2, mask = metrics._broadcast_masks(err2, mask)
        # components fixed by symmetry (zero spread and zero error) carry no information
        mask = jnp.broadcast_to(mask, err2.shape) & ((var > 0) | (err2 > 0))
        var = jnp.where(mask, var, 1.0)
        err2 = jnp.where(mask, err2, 1.0)

        def total(x):
            return jnp.where(mask, x, 0.0).sum()

        return cls(
            log_var=total(jnp.log(var)),
            z2=total(err2 / var),
            err2=total(err2),
            log_err2=total(jnp.log(jnp.maximum(err2, 1e-30))),
            count=mask.sum(),
        )

    def merge(self, other):
        return type(self)(
            self.log_var + other.log_var,
            self.z2 + other.z2,
            self.err2 + other.err2,
            self.log_err2 + other.log_err2,
            self.count + other.count,
        )

    def _nlls(self):
        n = self.count
        model = 0.5 * (self.log_var + self.z2) / n
        base = 0.5 * (jnp.log(self.err2 / n) + 1.0)
        oracle = 0.5 * (self.log_err2 / n + 1.0)
        alpha2 = self.z2 / n
        calibrated = 0.5 * (jnp.log(alpha2) + self.log_var / n + 1.0)
        return model, base, oracle, calibrated, alpha2


@flax.struct.dataclass
class NLL(GaussianScores):
    def compute(self):
        return self._nlls()[0] + 0.5 * np.log(2 * np.pi)


@flax.struct.dataclass
class RLL(GaussianScores):
    """Relative log-likelihood in % (Kellner & Ceriotti): 0 for a constant sigma
    equal to the RMSE, 100 for sigma = |error|."""

    def compute(self):
        model, base, oracle, _, _ = self._nlls()
        return 100 * (model - base) / (oracle - base)


@flax.struct.dataclass
class CalibratedRLL(GaussianScores):
    """RLL after rescaling all sigmas by the NLL-optimal factor alpha."""

    def compute(self):
        _, base, oracle, calibrated, _ = self._nlls()
        return 100 * (calibrated - base) / (oracle - base)


@flax.struct.dataclass
class CalibrationAlpha(GaussianScores):
    """NLL-optimal factor alpha for sigma -> alpha * sigma; 1 is calibrated."""

    def compute(self):
        return jnp.sqrt(self._nlls()[4])


def gaussian_score_fn(
    inputs: dict, label: dict[jnp.array], prediction: dict[jnp.array], key: str
) -> dict:
    err2 = (label[key] - prediction[key]) ** 2
    var = prediction[key + "_uncertainty"] ** 2
    return dict(err2=err2, var=var, mask=_mask(inputs, err2))


UNCERTAINTY_REDUCTIONS = {
    "nll": NLL,
    "rll": RLL,
    "rll_cal": CalibratedRLL,
    "calib_alpha": CalibrationAlpha,
}


def make_single_metric(key: str, reduction: str) -> metrics.Average:
    """
    Builds a single `clu` metric where the key picks out the quantity from
    the model predictions dict.
    Metric functions (like `mae_fn`) are curried with the `key`.
    """
    reduction_fns = {
        "mae": mae_fn,
        "mse": mse_fn,
        "rmse": mse_fn,
        "cosine_sim": cosine_sim,
        "per_atom_mae": per_atom_mae_fn,
        "per_atom_mse": per_atom_mse_fn,
        "per_atom_rmse": per_atom_mse_fn,
    }
    if reduction in UNCERTAINTY_REDUCTIONS:
        metric = UNCERTAINTY_REDUCTIONS[reduction]
        return metric.from_fun(partial(gaussian_score_fn, key=key))
    if reduction in ["rmse", "per_atom_rmse"]:
        metric = RootAverage
    else:
        metric = Averagefp64

    reduction_fn = reduction_fns[reduction]
    reduction_fn = partial(reduction_fn, key=key)

    return metric.from_fun(reduction_fn)


def initialize_metrics(metrics_list, uncertainties=None, strict=True) -> metrics.Collection:
    """
    Builds a `clu` metrics `Collection` by looping over all `keys` and `reductions`.
    the metrics are named according to `key_reduction`.
    See `make_single_metric` for details on the individual metrics.

    `uncertainties` is the set of quantities the model predicts an uncertainty for
    (None: assume all). Uncertainty metrics on other quantities raise an error if
    `strict`, otherwise they are left out (training metrics of a model that only
    computes the ensemble spread for validation).
    """
    log.info("Initializing Metrics")
    keys = []
    reductions = []
    for metric in metrics_list:
        for reduction in metric.reductions:
            unavailable = (
                reduction in UNCERTAINTY_REDUCTIONS
                and uncertainties is not None
                and metric.name not in uncertainties
            )
            if unavailable and strict:
                raise ValueError(
                    f"Metric '{metric.name}_{reduction}' needs '{metric.name}_uncertainty',"
                    " which this model does not predict"
                )
            if unavailable:
                log.info("Not computing %s_%s on the training set", metric.name, reduction)
                continue
            keys.append(metric.name)
            reductions.append(reduction)

    metric_dict = {}
    for key, reduction in zip(keys, reductions):
        metric = make_single_metric(key, reduction)
        metric_identifier = f"{key}_{reduction}"
        metric_dict[metric_identifier] = metric

    metrics_collection = metrics.Collection.create(**metric_dict)

    return metrics_collection
