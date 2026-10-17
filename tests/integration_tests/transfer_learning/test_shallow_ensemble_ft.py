import csv
import pathlib
import uuid

import numpy as np
import pytest
from ase.io import write
from flax.traverse_util import flatten_dict

from apax.train.checkpoints import restore_parameters
from tests.conftest import load_config_and_run_training

TEST_PATH = pathlib.Path(__file__).parent.resolve()
N_MEMBERS = 8


def _log(path):
    with open(path / "log.csv") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("num_data", (30,))
@pytest.mark.parametrize("force_loss", ("mse", "nll", "nll_3x3", "crps"))
def test_single_model_to_shallow_ensemble(get_tmp_path, example_dataset, force_loss):
    working_dir = get_tmp_path / str(uuid.uuid4())
    data_path = get_tmp_path / "ds.extxyz"
    write(data_path, example_dataset)

    data = {"directory": working_dir.as_posix(), "data_path": data_path.as_posix()}
    load_config_and_run_training(
        TEST_PATH / "config_base.yaml", {"data": {**data, "experiment": "base"}}
    )

    uq_metrics = ["mae", "nll", "rll", "rll_cal", "calib_alpha"]
    load_config_and_run_training(
        TEST_PATH / "config_ft.yaml",
        {
            "data": {**data, "experiment": "ens"},
            "model": {"ensemble": {"kind": "shallow", "n_members": N_MEMBERS}},
            "loss": [
                {"name": "energy", "loss_type": "nll"},
                {"name": "forces", "loss_type": force_loss},
            ],
            "metrics": [
                {"name": "energy", "reductions": uq_metrics},
                {"name": "forces", "reductions": uq_metrics},
            ],
            "transfer_learning": {
                "base_model_checkpoint": (working_dir / "base").as_posix(),
                "freeze_layers": [],
            },
        },
    )

    _, base = restore_parameters(working_dir / "base")
    _, ens = restore_parameters(working_dir / "ens")
    base, ens = flatten_dict(base, sep="/"), flatten_dict(ens, sep="/")
    out = [k for k in ens if k.endswith("readout/dense_2/w")]
    assert len(out) == 1
    (out,) = out
    assert base[out].shape[-1] == 1 and ens[out].shape[-1] == N_MEMBERS
    assert np.std(ens[out], axis=-1).min() > 0  # members differ
    # the backbone trained (NaN gradients are zeroed by the optimizer, which would
    # silently freeze it)
    hidden = [k for k in base if k.endswith("readout/dense_0/w")]
    assert hidden and not np.allclose(base[hidden[0]], ens[hidden[0]])

    rows = _log(working_dir / "ens")
    assert len(rows) == 2
    last = rows[-1]
    for q in ("energy", "forces"):
        for m in uq_metrics:
            v = float(last[f"val_{q}_{m}"])
            assert np.isfinite(v), (q, m, v)
    # the training step only has the per-member forces if the force loss uses them
    assert ("train_forces_rll" in last) == (force_loss != "mse")
    assert "train_energy_rll" in last

