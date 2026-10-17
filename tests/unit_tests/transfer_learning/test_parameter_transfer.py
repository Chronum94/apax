from apax.transfer_learning import black_list_param_transfer


def test_param_transfer():
    source = {
        "params": {
            "dense": {"w": 1, "b": 2},
            "basis": {"emb": 3},
        }
    }
    target = {
        "params": {
            "dense": {"w": 0, "b": 0},
            "basis": {"emb": 0},
        }
    }
    reinitialize_layers = ["basis"]
    transfered_target = black_list_param_transfer(source, target, reinitialize_layers)

    assert transfered_target["params"]["dense"]["w"] == source["params"]["dense"]["w"]
    assert transfered_target["params"]["dense"]["b"] == source["params"]["dense"]["b"]
    assert transfered_target["params"]["basis"]["emb"] == target["params"]["basis"]["emb"]


def test_single_output_layer_expanded_to_members():
    import numpy as np
    import pytest

    rng = np.random.default_rng(0)
    w = rng.normal(size=(8, 1))
    source = {"params": {"readout": {"dense_1": {"w": w}, "dense_0": {"w": np.ones((3, 8))}}}}
    target = {
        "params": {
            "readout": {"dense_1": {"w": np.zeros((8, 16))}, "dense_0": {"w": np.zeros((3, 8))}}
        }
    }
    out = black_list_param_transfer(source, target, [], ensemble_init_noise=0.05, seed=1)
    members = np.asarray(out["params"]["readout"]["dense_1"]["w"])
    assert members.shape == (8, 16)
    dev = members - w
    rms = np.sqrt(np.mean(w**2))
    assert 0.03 * rms < dev.std() < 0.07 * rms  # noise of the requested size
    assert np.abs(dev.mean(axis=1)).max() < 4 * 0.05 * rms / np.sqrt(16)  # centred on w
    np.testing.assert_array_equal(out["params"]["readout"]["dense_0"]["w"], np.ones((3, 8)))

    target["params"]["readout"]["dense_0"]["w"] = np.zeros((3, 9))
    with pytest.raises(ValueError, match="dense_0"):
        black_list_param_transfer(source, target, [])
