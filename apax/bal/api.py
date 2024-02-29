from functools import partial
from typing import List, Union

import jax
import numpy as np
from ase import Atoms
from click import Path
from tqdm import trange
from flax.core.frozen_dict import FrozenDict

from apax.bal import feature_maps, kernel, selection, transforms
from apax.data.input_pipeline import AtomisticDataset
from apax.model.builder import ModelBuilder
from apax.model.gmnn import EnergyModel
from apax.train.checkpoints import (
    canonicalize_energy_model_parameters,
    check_for_ensemble,
    restore_parameters,
)
from apax.train.run import initialize_dataset


def create_feature_fn(
    model: EnergyModel,
    params: FrozenDict,
    base_feature_map: feature_maps.FeatureTransformation,
    feature_transforms=[],
    is_ensemble: bool = False,
):
    """
    Converts a model into a feature map and transforms it as needed and
    sets it up for use in copmuting the features of a dataset.

    All transformations are applied on the feature function, not on computed features.
    Only the final function is jit compiled.


    Attributes
    ----------
    model: EnergyModel
        Model to be transformed.
    params: FrozenDict
        Model parameters
    base_feature_map: FeatureTransformation
        Class that transforms the model into a `FeatureMap`
    feature_transforms: list
        Feature tranforms to be applied on top of the base feature map transform.
        Examples would include multiplcation with or addition of a constant.
    is_ensemble: bool
        Whether or not to apply the ensemble transformation i.e. an averaging of kernels for model ensembles.
    """
    feature_fn = base_feature_map.apply(model)

    if is_ensemble:
        feature_fn = transforms.ensemble_features(feature_fn)

    for transform in feature_transforms:
        feature_fn = transform.apply(feature_fn)

    feature_fn = transforms.batch_features(feature_fn)
    feature_fn = partial(feature_fn, params)
    feature_fn = jax.jit(feature_fn)
    return feature_fn


def compute_features(feature_fn: feature_maps.FeatureMap, dataset: AtomisticDataset) -> np.ndarray:
    """Compute the features of a dataset.
    
    Attributes
    ----------
    feature_fn: FeatureMap
        Function to compute the features with.
    dataset: AtomisticDataset
        Dataset to compute the features for.
    """
    features = []
    n_data = dataset.n_data
    ds = dataset.batch()

    pbar = trange(n_data, desc="Computing features", ncols=100, leave=True)
    for inputs in ds:
        g = feature_fn(inputs)
        features.append(np.asarray(g))
        pbar.update(g.shape[0])
    pbar.close()

    features = np.concatenate(features, axis=0)
    return features


def kernel_selection(
    model_dir: Union[Path, List[Path]],
    train_atoms: List[Atoms],
    pool_atoms: List[Atoms],
    base_fm_options: dict,
    selection_method: str,
    feature_transforms: list = [],
    selection_batch_size: int = 10,
    processing_batch_size: int = 64,
) -> list[int]:
    """
    Main fuinction to facilitate batch data selection.
    Currently only the last layer gradient features and MaxDist selection method are available.
    More can be added as needed as this function is agnostic of the feature map/selection method internals.

    Attributes
    ----------
    model_dir: Union[Path, List[Path]]
        Path to the trained model or models which should be used to compute features.
    train_atoms: List[Atoms]
        List of `ase.Atoms` used to train the models.
    pool_atoms: List[Atoms]
        List of `ase.Atoms` to select new data from.
    base_fm_options:
        Dict
    selection_method:
    feature_transforms:
    selection_batch_size:
        Amount of new data points to be selected from `pool_atoms`.
    processing_batch_size:
        Amount of data points to compute the features for at once.
        Does not effect results, just the speed of processing.
    """
    selection_fn = {
        "max_dist": selection.max_dist_selection,
    }[selection_method]

    base_feature_map = feature_maps.FeatureMapOptions(base_fm_options)

    config, params = restore_parameters(model_dir)
    params = canonicalize_energy_model_parameters(params)
    n_models = check_for_ensemble(params)
    is_ensemble = n_models > 1

    n_train = len(train_atoms)
    dataset = initialize_dataset(
        config, train_atoms + pool_atoms, read_labels=False, calc_stats=False
    )
    dataset.set_batch_size(processing_batch_size)

    _, init_box = dataset.init_input()

    builder = ModelBuilder(config.model.get_dict(), n_species=119)
    model = builder.build_energy_model(apply_mask=True, init_box=init_box)

    feature_fn = create_feature_fn(
        model, params, base_feature_map, feature_transforms, is_ensemble
    )
    g = compute_features(feature_fn, dataset)
    km = kernel.KernelMatrix(g, n_train)
    new_indices = selection_fn(km, selection_batch_size)

    return new_indices
