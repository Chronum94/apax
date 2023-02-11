import pytest
import tensorflow as tf
import numpy as np

from gmnn_jax.data.input_pipeline import InputPipeline, pad_to_largest_element


@pytest.mark.parametrize(
    "num_data, pbc, calc_results, external_labels",
    (
        [5, False, ["energy"], None],
        [5, False, ["energy", "forces"], None],
        [5, True, ["energy", "forces"], None],
        [5, True, ["energy", "forces"], {"fixed": {"ma_tensors": np.random.uniform(low=-1.0, high=1.0, size=(5, 3, 3))}}],
    ),
)
def test_input_pipeline(example_atoms, pbc, calc_results, num_data, external_labels):
    batch_size = 2

    ds = InputPipeline(
        cutoff=6.0, batch_size=batch_size, atoms_list=example_atoms, n_epoch=1, external_labels=external_labels
    )
    assert ds.steps_per_epoch() == num_data // batch_size

    ds = ds.shuffle_and_batch()

    sample_inputs, sample_labels = next(ds)

    if pbc:
        assert "cell" in sample_inputs
        assert len(sample_inputs["cell"]) == batch_size
        assert len(sample_inputs["cell"][0]) == 3
    else:
        assert "cell" not in sample_inputs

    assert "numbers" in sample_inputs
    for i in range(batch_size):
        assert len(sample_inputs["numbers"][i]) == max(sample_inputs["n_atoms"])

    assert "idx" in sample_inputs
    assert len(sample_inputs["idx"][0]) == len(sample_inputs["idx"][1])

    assert "positions" in sample_inputs
    assert len(sample_inputs["positions"][0][0]) == 3
    for i in range(batch_size):
        assert len(sample_inputs["positions"][i]) == max(sample_inputs["n_atoms"])

    assert "n_atoms" in sample_inputs
    assert len(sample_inputs["n_atoms"]) == batch_size

    assert "energy" in sample_labels
    assert len(sample_labels["energy"]) == batch_size

    if 'forces' in calc_results:
        assert "forces" in sample_labels
        assert len(sample_labels["forces"][0][0]) == 3
        for i in range(batch_size):
            assert len(sample_labels["forces"][i]) == max(sample_inputs["n_atoms"])

    if external_labels:
        assert 'ma_tensors' in sample_labels
        assert len(sample_labels["ma_tensors"]) == batch_size

    sample_inputs2, _ = next(ds)
    assert (sample_inputs["positions"][0][0] != sample_inputs2["positions"][0][0]).all()

def test_pad_to_largest_element():
    r_inp = {"idx": tf.ragged.constant([[1, 4, 3], [4, 5, 2, 3, 1]])}
    p_inp = {"n_atoms": tf.constant([3, 5])}
    r_lab = {"forces": tf.ragged.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0, 3.0]])}
    p_lab = {"energy": tf.constant([103.3, 98.4])}

    inputs, labels = pad_to_largest_element(r_inp, p_inp, r_lab, p_lab)

    assert "idx" in inputs
    assert len(inputs["idx"][0]) == len(inputs["idx"][1])

    assert "n_atoms" in inputs

    assert "forces" in labels
    assert len(labels["forces"][0]) == len(labels["forces"][1])

    assert "energy" in labels