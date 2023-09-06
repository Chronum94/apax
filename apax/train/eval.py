import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from tqdm import trange

from apax.config import parse_train_config
from apax.data.statistics import compute_scale_shift_parameters
from apax.model import ModelBuilder
from apax.train.checkpoints import load_params
from apax.train.metrics import initialize_metrics
from apax.train.run import (
    RawDataset,
    initialize_callbacks,
    initialize_dataset,
    initialize_loss_fn,
    setup_logging,
)
from apax.train.trainer import make_step_fns
from apax.utils.data import load_data, split_atoms, split_label
from apax.utils.random import seed_py_np_tf

log = logging.getLogger(__name__)


def get_test_idxs(atoms_list, used_idxs, n_test=-1):
    idxs = np.arange(len(atoms_list))
    test_idxs = np.setdiff1d(idxs, used_idxs)
    np.random.shuffle(test_idxs)
    if n_test != -1:
        test_idxs = test_idxs[:n_test]

    return test_idxs


def load_test_data(
    config, model_version_path, eval_path, n_test=-1
):  # TODO double code run.py in progress
    log.info("Running Input Pipeline")
    os.makedirs(eval_path, exist_ok=True)
    if config.data.data_path is not None:
        log.info(f"Read data file {config.data.data_path}")
        atoms_list, label_dict = load_data(config.data.data_path)

        idxs_dict = np.load(model_version_path / "train_val_idxs.npz")

        used_idxs = idxs_dict["train_idxs"]
        np.append(used_idxs, idxs_dict["val_idxs"])

        test_idxs = get_test_idxs(atoms_list, used_idxs, n_test)

        np.savez(
            os.path.join(eval_path, "test_idxs"),
            test_idxs=test_idxs,
        )

        atoms_list, _ = split_atoms(atoms_list, test_idxs)
        label_dict, _ = split_label(label_dict, test_idxs)

    elif config.data.test_data_path is not None:
        log.info(f"Read test data file {config.data.test_data_path}")
        atoms_list, label_dict = load_data(config.data.test_data_path)
        atoms_list = atoms_list[:n_test]
        for key, val in label_dict.items():
            label_dict[key] = val[:n_test]
    else:
        raise ValueError("input data path/paths not defined")

    test_raw_ds = RawDataset(atoms_list=atoms_list, additional_labels=label_dict)
    return test_raw_ds


def load_params(model_version_path):
    best_dir = model_version_path / "best"
    log.info(f"load checkpoint from {best_dir}")
    try:
        raw_restored = checkpoints.restore_checkpoint(best_dir, target=None, step=None)
    except FileNotFoundError:
        print(f"No checkpoint found at {best_dir}")
    params = jax.tree_map(jnp.asarray, raw_restored["model"]["params"])

    return params

def predict(model, params, Metrics, loss_fn, test_ds, callbacks):
    callbacks.on_train_begin()
    _, test_step_fn = make_step_fns(loss_fn, Metrics, model=model, sam_rho=0.0)

    test_steps_per_epoch = test_ds.steps_per_epoch()
    batch_test_ds = test_ds.shuffle_and_batch()

    epoch_loss = {}
    epoch_start_time = time.time()

    epoch_loss.update({"test_loss": 0.0})
    test_metrics = Metrics.empty()
    with trange(
        0, test_steps_per_epoch, desc="Batches", ncols=100, disable=False, leave=True
    ) as batch_pbar:
        for batch_idx in range(test_steps_per_epoch):
            inputs, labels = next(batch_test_ds)

            test_metrics, batch_loss = test_step_fn(params, inputs, labels, test_metrics)

            epoch_loss["test_loss"] += batch_loss
            batch_pbar.set_postfix(test_loss=epoch_loss["test_loss"] / batch_idx)
            batch_pbar.update()

    epoch_loss["test_loss"] /= test_steps_per_epoch
    epoch_loss["test_loss"] = float(epoch_loss["test_loss"])
    epoch_metrics = {
        f"test_{key}": float(val) for key, val in test_metrics.compute().items()
    }
    epoch_metrics.update({**epoch_loss})
    epoch_end_time = time.time()
    epoch_metrics.update({"epoch_time": epoch_end_time - epoch_start_time})
    callbacks.on_epoch_end(epoch=1, logs=epoch_metrics)
    callbacks.on_train_end()
    # TODO currently this has no informative output


def eval_model(config_path, n_test=-1, log_file="eval.log", log_level="error"):
    setup_logging(log_file, log_level)
    log.info("Starting model evaluation")
    config = parse_train_config(config_path)

    seed_py_np_tf(config.seed)

    model_version_path = Path(config.data.directory) / config.data.experiment
    eval_path = model_version_path / "eval"

    callbacks = initialize_callbacks(config.callbacks, eval_path)
    loss_fn = initialize_loss_fn(config.loss)
    Metrics = initialize_metrics(config.metrics)

    test_raw_ds = load_test_data(config, model_version_path, eval_path, n_test)

    test_ds = initialize_dataset(config, test_raw_ds)
    ds_stats = compute_scale_shift_parameters(
        test_raw_ds.atoms_list,
        config.data.shift_method,
        config.data.scale_method,
        config.data.shift_options,
        config.data.scale_options,
    )

    init_input = test_ds.init_input()
    init_box = np.array(init_input["box"][0])

    builder = ModelBuilder(config.model.get_dict(), n_species=ds_stats.n_species)
    model = builder.build_energy_derivative_model(
        scale=ds_stats.elemental_scale,
        shift=ds_stats.elemental_shift,
        apply_mask=True,
        init_box=init_box,
    )

    model = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0))

    params = load_params(model_version_path)

    predict(model, params, Metrics, loss_fn, test_ds, callbacks)
