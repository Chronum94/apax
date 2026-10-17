import jax.numpy as jnp
import optax
from flax.core.frozen_dict import freeze
from jax import tree_util

from apax.optimizer import get_opt


def test_get_opt():
    params = {
        "my_linear": {"w": jnp.ones((3, 3)), "b": jnp.ones((3,))},
        "my_embed": {
            "atomic_type_embedding": jnp.ones((3, 3)),
        },
        "my_scale_shift": {
            "scale_per_element": jnp.ones((3,)),
            "shift_per_element": jnp.ones((3,)),
        },
    }
    params = freeze(params)

    grads = tree_util.tree_map(lambda x: x * 0.01, tree=params)

    opt = get_opt(
        params,
        0,
        500,
        emb_lr=0.05,
        nn_lr=0.01,
        scale_lr=0.001,
        shift_lr=0.1,
        schedule={"name": "linear", "transition_begin": 0, "end_value": 1e-6},
    )
    opt_state = opt.init(params=params)

    updates, new_opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    filtered_params = tree_util.tree_map(
        lambda v: v[0] if len(v.shape) == 1 else v[0, 0], new_params
    )
    flat_params, treedef = tree_util.tree_flatten(filtered_params)

    assert jnp.allclose(flat_params[1], flat_params[2])
    assert flat_params[-2] > flat_params[1]
    assert flat_params[0] < flat_params[1]
    assert flat_params[-1] < flat_params[1]


def test_muon_keeps_readout_output_layers_on_adam():
    import numpy as np

    from apax.optimizer.get_optimizer import readout_output_layers

    rng = np.random.default_rng(0)
    params = freeze(
        {
            "energy_model": {
                "readout": {
                    "dense_0": {"w": jnp.asarray(rng.normal(size=(6, 8)))},
                    "dense_1": {"w": jnp.asarray(rng.normal(size=(8, 8)))},
                    "dense_2": {"w": jnp.asarray(rng.normal(size=(8, 4)))},  # 4 members
                },
                "head": {"readout": {"dense_0": {"w": jnp.ones((8, 1))}}},
            }
        }
    )
    assert readout_output_layers(params) == {
        ("energy_model", "readout", "dense_2"),
        ("energy_model", "head", "readout", "dense_0"),
    }

    nn_lr = 0.01
    opt = get_opt(
        params, 1, 10, nn_lr=nn_lr, name="muon", kwargs={"muon_lr": 0.02},
        schedule={"name": "linear", "transition_begin": 0, "end_value": 1e-6},
    )
    grads = tree_util.tree_map(lambda x: jnp.asarray(rng.normal(size=x.shape)), params)
    updates, _ = opt.update(grads, opt.init(params), params)
    out = np.abs(np.asarray(updates["energy_model"]["readout"]["dense_2"]["w"]))
    hidden = np.abs(np.asarray(updates["energy_model"]["readout"]["dense_1"]["w"]))
    # Adam's first step is lr * sign(g) per element; Muon's is not
    np.testing.assert_allclose(out, nn_lr, rtol=1e-3)
    assert not np.allclose(hidden, hidden.flat[0], rtol=1e-2)
