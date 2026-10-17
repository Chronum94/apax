import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from apax.nn.models import _member_std
from apax.train.loss import crps_ensemble_loss, crps_loss, nll_3x3, nll_loss
from apax.utils.math import inv_and_det_3x3


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def _spd(rng, scale, n=50):
    A = rng.normal(size=(n, 3, 3)) * scale
    return A @ np.swapaxes(A, 1, 2) + 0.1 * scale**2 * np.eye(3)


@pytest.mark.parametrize("scale", [1.0, 0.1, 0.02, 1e-3])
def test_inv_and_det_3x3(scale):
    S = _spd(np.random.default_rng(0), scale)
    inv, det = inv_and_det_3x3(jnp.asarray(S))
    np.testing.assert_allclose(inv, np.linalg.inv(S), rtol=1e-9)
    np.testing.assert_allclose(det, np.linalg.det(S), rtol=1e-9)


def _ensemble(rng, sigma, n_atoms=20, K=32):
    """Force ensemble (batch 1, n_atoms, 3, K) with anisotropic, correlated spread."""
    A = rng.normal(size=(n_atoms, 3, 3)) * sigma
    ens = np.einsum("aij,ajk->aik", A, rng.normal(size=(n_atoms, 3, K)))
    ens += rng.normal(size=(n_atoms, 3, 1))
    return ens[None]


@pytest.mark.parametrize("sigma", [0.3, 0.05, 0.01])
def test_nll_3x3_matches_multivariate_normal(sigma):
    rng = np.random.default_rng(1)
    ens = _ensemble(rng, sigma)
    mean = ens.mean(-1)
    label = mean + rng.normal(size=mean.shape) * sigma
    pred = {"forces": jnp.asarray(mean), "forces_ensemble": jnp.asarray(ens)}
    got = np.asarray(nll_3x3({"forces": jnp.asarray(label)}, pred, "forces"))[0]

    ref = []
    for a in range(ens.shape[1]):
        cov = np.cov(ens[0, a], ddof=1)
        cov += 1e-4 * np.trace(cov) / 3 * np.eye(3) + 1e-12 * np.eye(3)
        lp = stats.multivariate_normal(mean[0, a], cov).logpdf(label[0, a])
        ref.append(-lp - 1.5 * np.log(2 * np.pi))  # loss leaves out the constant
    np.testing.assert_allclose(got, ref, rtol=1e-8)


def test_nll_and_crps_match_reference():
    rng = np.random.default_rng(2)
    mu, sigma = rng.normal(size=20), rng.uniform(0.01, 1.0, size=20)
    y = mu + sigma * rng.normal(size=20)
    pred = {"x": jnp.asarray(mu), "x_uncertainty": jnp.asarray(sigma)}
    lab = {"x": jnp.asarray(y)}

    nll = np.asarray(nll_loss(lab, pred, "x"))
    ref = -stats.norm(mu, sigma).logpdf(y) - 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(nll, ref, rtol=1e-6)

    crps = np.asarray(crps_loss(lab, pred, "x"))
    for i in range(5):  # CRPS = int (F(x) - 1[x >= y])^2 dx
        F = stats.norm(mu[i], sigma[i]).cdf
        lo, hi = mu[i] - 12 * sigma[i], mu[i] + 12 * sigma[i]
        left = integrate.quad(lambda x: F(x) ** 2, lo, y[i])[0]
        right = integrate.quad(lambda x: (F(x) - 1) ** 2, y[i], hi)[0]
        np.testing.assert_allclose(crps[i], left + right, rtol=1e-6)


@pytest.mark.parametrize("loss", [nll_loss, crps_loss, nll_3x3, crps_ensemble_loss])
def test_finite_gradient_with_zero_spread(loss):
    """Padded atoms and components fixed by symmetry have members that agree exactly."""

    def f(w):
        # 4 members x 2 atoms x 3; atom 1 identical in all members, atom 0's z too
        base = jnp.array([[0.1, -0.2, 0.0], [0.0, 0.0, 0.0]])
        spread = jnp.array([[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        ens = base + w * jnp.array([1.0, -1.0, 0.5, -0.5])[:, None, None] * spread
        mean = ens.mean(0)
        pred = {
            "forces": mean[None],
            "forces_uncertainty": _member_std(ens)[None],
            "forces_ensemble": jnp.transpose(ens, (1, 2, 0))[None],
        }
        label = {"forces": jnp.array([[[0.3, 0.1, 0.0], [0.0, 0.0, 0.0]]])}
        return loss(label, pred, "forces").sum()

    g = jax.grad(f)(0.2)
    assert np.isfinite(g) and g != 0


def test_crps_ensemble():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4, 5, 8))  # members last
    y = rng.normal(size=(4, 5))
    got = np.asarray(
        crps_ensemble_loss({"x": jnp.asarray(y)}, {"x_ensemble": jnp.asarray(X)}, "x")
    )
    K = X.shape[-1]
    ref = np.abs(X - y[..., None]).mean(-1) - np.abs(
        X[..., :, None] - X[..., None, :]
    ).sum((-2, -1)) / (2 * K * (K - 1))
    np.testing.assert_allclose(got, ref, rtol=1e-12)

    # unbiased: for members drawn from N(mu, s^2) its mean is the Gaussian CRPS
    mu, sig, yy = 0.3, 0.7, -0.2
    draws = mu + sig * rng.normal(size=(20000, 4))
    fair = crps_ensemble_loss(
        {"x": jnp.full(20000, yy)}, {"x_ensemble": jnp.asarray(draws)}, "x"
    )
    gauss = crps_loss(
        {"x": jnp.array([yy])}, {"x": jnp.array([mu]), "x_uncertainty": jnp.array([sig])}, "x"
    )
    np.testing.assert_allclose(float(jnp.mean(fair)), float(gauss[0]), rtol=0.02)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_beta_nll_mean_gradient(beta):
    """d/dmu of beta-NLL = (mu - y) / sigma^(2 - 2 beta); beta = 1 is the MSE gradient."""
    mu, sig, y = 0.3, 0.2, -0.1

    def f(m):
        pred = {"x": jnp.array([m]), "x_uncertainty": jnp.array([sig])}
        return nll_loss({"x": jnp.array([y])}, pred, "x", {"beta": beta}).sum()

    var = sig**2 + 1e-12
    np.testing.assert_allclose(jax.grad(f)(mu), (mu - y) / var ** (1 - beta), rtol=1e-6)
