import jax
import jax.numpy as jnp
import numpy as np
import optax
from clu import metrics
from flax import struct
from flax.training import train_state

from apax.train.trainer import make_scan_epoch, make_step_fns


def toy_model(params, R, Z, idx, box, offsets):
    return {"energy": jnp.sum(params["w"] * R**2, axis=(1, 2)) + params["b"]}


def toy_loss(inputs, predictions, labels):
    return jnp.mean((predictions["energy"] - labels["energy"]) ** 2)


@struct.dataclass
class ToyMetrics(metrics.Collection):
    err: metrics.Average.from_fun(
        lambda inputs, label, prediction: jnp.abs(prediction["energy"] - label["energy"])
    )


def test_scan_epoch_matches_step_loop():
    """One scanned epoch == the step loop over the same permutation (params, average, loss, metrics)."""
    n, bs, steps, decay = 10, 3, 3, 0.9
    rng = np.random.default_rng(0)
    inputs = {
        "positions": jnp.asarray(rng.normal(size=(n, 4, 3))),
        "numbers": jnp.ones((n, 4), jnp.int16),
        "idx": jnp.zeros((n, 2, 6), jnp.int16),
        "box": jnp.zeros((n, 3, 3)),
        "offsets": jnp.zeros((n, 6, 3)),
    }
    data = (inputs, {"energy": jnp.asarray(rng.normal(size=n))})
    train_step, _ = make_step_fns(toy_loss, ToyMetrics, toy_model, is_ensemble=False)

    def fresh():
        p = {"w": jnp.array(0.3), "b": jnp.array(0.1)}
        return train_state.TrainState.create(apply_fn=toy_model, params=p, tx=optax.adam(0.05))

    key = jax.random.PRNGKey(3)
    state, ema, m, loss_sum = make_scan_epoch(train_step, decay, steps, bs)(
        fresh(), jax.tree.map(jnp.copy, fresh().params), ToyMetrics.empty(), data, key, True
    )

    ref, ref_ema, ref_m, ref_loss = fresh(), fresh().params, ToyMetrics.empty(), 0.0
    for idx in jax.random.permutation(key, n)[: steps * bs].reshape(steps, bs):
        batch = jax.tree.map(lambda x: x[idx], data)
        (ref, ref_m), loss = train_step((ref, ref_m), batch)
        ref_ema = jax.tree.map(lambda p, a: decay * a + (1 - decay) * p, ref.params, ref_ema)
        ref_loss += float(loss)

    for k in ("w", "b"):
        np.testing.assert_allclose(state.params[k], ref.params[k], rtol=1e-12)
        np.testing.assert_allclose(ema[k], ref_ema[k], rtol=1e-12)
    np.testing.assert_allclose(loss_sum, ref_loss, rtol=1e-12)
    np.testing.assert_allclose(m.compute()["err"], ref_m.compute()["err"], rtol=1e-12)
