import logging
from typing import Union

import jax.numpy as jnp
import numpy as np

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict

from apax.config.train_config import TransferLearningConfig
from apax.train.checkpoints import load_params

log = logging.getLogger(__name__)


def expand_to_members(source, target_shape, noise, rng):
    """A single output column (..., 1) copied to every member (..., n_members), plus
    isotropic Gaussian noise of `noise` times the RMS of the source column."""
    source = np.asarray(source)
    tiled = np.broadcast_to(source, target_shape)
    rms = np.sqrt(np.mean(source**2))
    return tiled + noise * rms * rng.standard_normal(target_shape)


def black_list_param_transfer(
    source_params: Union[FrozenDict, dict],
    target_params: Union[FrozenDict, dict],
    param_black_list: list[str],
    ensemble_init_noise: float = 0.1,
    seed: int = 0,
) -> FrozenDict:
    """Transfer parameters from one dictionary to another, while keeping
        some key-value pairs unchanged.

    A source parameter with one output column whose target has several (a single
    model's readout loaded into a shallow ensemble) is copied to every member, with
    noise (`ensemble_init_noise` times its RMS) so that the members differ. Any other
    shape mismatch is an error.

    Args:
        source_params (Union[FrozenDict, dict]): source parameters
        target_params (Union[FrozenDict, dict]): target parameters
        param_black_list (list[str]): list of keys to keep unchanged.
        ensemble_init_noise (float): relative noise for expanded output layers.
        seed (int): seed of that noise.

    Returns:
        transfered_target (dict): target_params with key-value pairs updated.

    """
    source_params = unfreeze(source_params)
    target_params = unfreeze(target_params)

    flat_source = flatten_dict(source_params)
    flat_target = flatten_dict(target_params)
    rng = np.random.default_rng(seed)
    for p, v in flat_source.items():
        if p[-2] in param_black_list:
            continue
        if p not in flat_target:
            log.warning("Skipping %s: not a parameter of the target model", "/".join(p))
            continue
        source_shape, target_shape = np.shape(v), np.shape(flat_target[p])
        if source_shape == target_shape:
            flat_target[p] = v
            log.info("Transferring parameter: %s", p[-2])
        elif (
            len(source_shape) == len(target_shape) > 0
            and source_shape[:-1] == target_shape[:-1]
            and source_shape[-1] == 1
        ):
            flat_target[p] = jnp.asarray(
                expand_to_members(v, target_shape, ensemble_init_noise, rng),
                dtype=flat_target[p].dtype,
            )
            log.info(
                "Expanding %s from 1 to %d ensemble members (noise %.3g x RMS)",
                "/".join(p),
                target_shape[-1],
                ensemble_init_noise,
            )
        else:
            raise ValueError(
                f"Cannot transfer {'/'.join(p)}: shape {source_shape} vs {target_shape}"
            )

    transfered_target = unflatten_dict(flat_target)
    transfered_target = freeze(transfered_target)
    return transfered_target


def transfer_parameters(
    state: TrainState, ckpt_config: TransferLearningConfig, seed: int = 0
) -> TrainState:
    """Transfer the parameters from the checkpoint to the train state.

    Args:
        state (TrainState): train state
        ckpt_config (TransferLearningConfig): transfer learning configuration

    Returns:
        state (TrainState): TrainState with the `params` attribute updated
            according to the transfer learning configuration.
    """
    source_params = load_params(ckpt_config.base_model_checkpoint)
    log.info("Transferring parameters from %s", ckpt_config.base_model_checkpoint)
    params = black_list_param_transfer(
        source_params,
        state.params,
        ckpt_config.reset_layers,
        ckpt_config.ensemble_init_noise,
        seed,
    )
    state = state.replace(params=params)
    return state
