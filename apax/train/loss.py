import dataclasses
from typing import List

import jax
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np
from ase.data import atomic_masses

from apax.utils.math import inv_and_det_3x3, normed_dotp


def weighted_squared_error(
    label: jnp.array,
    prediction: jnp.array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
) -> jnp.array:
    """
    Squared error function that allows weighting of
    individual contributions by the number of atoms in the system.
    """
    label, prediction = label[name], prediction[name]
    return (label - prediction) ** 2


def weighted_huber_loss(
    label: jnp.array,
    prediction: jnp.array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
) -> jnp.array:
    """
    Huber loss function that allows weighting of
    individual contributions by the number of atoms in the system.
    """
    label, prediction = label[name], prediction[name]
    if "delta" not in parameters.keys():
        raise KeyError("Huber loss function requires 'delta' parameter")
    delta = parameters["delta"]
    diff = jnp.abs(label - prediction)
    loss = jnp.where(diff > delta, delta * (diff - 0.5 * delta), 0.5 * diff**2)
    return loss


def pairwise_squared_error(
    label: jnp.array,
    prediction: jnp.array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
) -> jnp.array:
    """Squared error of within-batch differences of the per-atom error d_i = (pred_i - label_i) / n_atoms_i,
    averaged over all pairs (i, j) of the batch: mean_{i<j} (d_i - d_j)^2. Returned per structure as
    2N/(N-1) * (d_i - mean d)^2, whose batch mean equals the all-pairs mean. A batch-common error offset is
    invisible to it, so it weights relative energies within a batch. Use with atoms_exponent 0."""
    d = (prediction[name] - label[name]) / inputs["n_atoms"]
    n = d.shape[0]
    return 2.0 * n / max(n - 1, 1) * (d - jnp.mean(d)) ** 2


def _floored_sigma(sigmas, parameters):
    """sqrt(sigma^2 + eps^2): a smooth floor (default eps 1e-6, in the units of the
    quantity). A hard clip has zero gradient below the floor."""
    eps = parameters.get("eps", 1e-6)
    return jnp.sqrt(sigmas**2 + eps**2)


def crps_loss(
    label: jax.Array,
    prediction: jax.Array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
) -> jax.Array:
    """Computes the CRPS of a gaussian distribution given
    means, targets and standard deviations (uncertainty estimate)
    """
    label = label[name]
    means = prediction[name]
    sigmas = _floored_sigma(prediction[name + "_uncertainty"], parameters)

    norm_x = (label - means) / sigmas
    cdf = 0.5 * (1 + jsc.special.erf(norm_x / jnp.sqrt(2)))

    normalization = 1 / (jnp.sqrt(2.0 * np.pi))

    pdf = normalization * jnp.exp(-(norm_x**2) / 2.0)

    crps = sigmas * (norm_x * (2 * cdf - 1) + 2 * pdf - 1 / jnp.sqrt(np.pi))

    return crps


def crps_ensemble_loss(
    label: jax.Array,
    prediction: jax.Array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
) -> jax.Array:
    """Fair (unbiased) CRPS of the ensemble members themselves, without a Gaussian
    assumption: mean_k |x_k - y| - sum_{k,l} |x_k - x_l| / (2 K (K - 1)).
    Pairwise rather than sort-based (XLA's GPU sort is unreliable on some ROCm GPUs).
    """
    y = label[name][..., None]
    members = prediction[name + "_ensemble"]  # members on the last axis
    K = members.shape[-1]
    skill = jnp.mean(jnp.abs(members - y), axis=-1)
    pairwise = jnp.abs(members[..., :, None] - members[..., None, :])
    spread = jnp.sum(pairwise, axis=(-2, -1)) / (2 * K * (K - 1))
    return skill - spread


def nll_loss(
    label: jax.Array,
    prediction: jax.Array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
) -> jax.Array:
    """Computes the gaussian NLL loss given
    means, targets and standard deviations (uncertainty estimate).

    parameters["beta"] (default 0): beta-NLL (Seitzer et al., ICLR 2022), each point's NLL
    weighted by stop_gradient(sigma^(2 beta)). beta = 1 gives the mean exactly the MSE
    gradient (no down-weighting of uncertain points); 0.5 weights by 1/sigma.
    """
    label = label[name]
    means = prediction[name]
    variances = _floored_sigma(prediction[name + "_uncertainty"], parameters) ** 2

    x1 = jnp.log(variances)
    x2 = ((means - label) ** 2) / variances
    nll = 0.5 * (x1 + x2)

    beta = parameters.get("beta", 0.0)
    if beta:
        nll = nll * jax.lax.stop_gradient(variances**beta)
    return nll


def force_angle_loss(
    label: jnp.array,
    prediction: jnp.array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
) -> jnp.array:
    """
    Consine similarity loss function. Contributions are summed in `Loss`.
    """
    label, prediction = label[name], prediction[name]
    dotp = normed_dotp(label, prediction)
    return 1.0 - dotp


def force_angle_div_force_label(
    label: jnp.array,
    prediction: jnp.array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
):
    """
    Consine similarity loss function weighted by the norm of the force labels.
    Contributions are summed in `Loss`.
    """
    label, prediction = label[name], prediction[name]
    dotp = normed_dotp(label, prediction)
    F_0_norm = jnp.linalg.norm(label, ord=2, axis=2, keepdims=False)
    loss = jnp.where(F_0_norm > 1e-6, (1.0 - dotp) / F_0_norm, jnp.zeros_like(dotp))
    return loss


def force_angle_exponential_weight(
    label: jnp.array,
    prediction: jnp.array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
) -> jnp.array:
    """
    Consine similarity loss function exponentially scaled by the norm of the force labels.
    Contributions are summed in `Loss`.
    """
    label, prediction = label[name], prediction[name]
    dotp = normed_dotp(label, prediction)
    F_0_norm = jnp.linalg.norm(label, ord=2, axis=2, keepdims=False)
    return (1.0 - dotp) * jnp.exp(-F_0_norm)


def stress_tril(
    label: jnp.array,
    prediction: jnp.array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
):
    label, prediction = label[name], prediction[name]
    idxs = jnp.tril_indices(3)
    label_tril = label[:, idxs[0], idxs[1]]
    prediction_tril = prediction[:, idxs[0], idxs[1]]
    return (label_tril - prediction_tril) ** 2


def nll_3x3(
    label: jnp.array,
    prediction: jnp.array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
):
    label = label[name]
    means = prediction[name]
    ensemble = prediction[name + "_ensemble"]

    diff = label - means

    deviations = ensemble - means[..., None]
    # K = deviations.shape[2]  # Number of members
    K = deviations.shape[-1]  # Number of ensemble members
    if K < 2:
        raise ValueError("nll_3x3 requires at least 2 ensemble members")
    Sigma = jnp.einsum("bijk,bilk->bijl", deviations, deviations) / (
        K - 1
    )  # Sample covariance matrix

    # Jitter relative to the covariance's own scale (bounds the condition number at
    # ~1/rel_jitter whatever the units or force magnitude), plus a tiny absolute
    # floor so that a zero covariance (padded atoms) stays invertible.
    rel_jitter = parameters.get("rel_jitter", 1e-4)
    abs_jitter = parameters.get("abs_jitter", 1e-12)
    scale = jnp.trace(Sigma, axis1=-2, axis2=-1)[..., None, None] / 3
    Sigma = Sigma + (rel_jitter * scale + abs_jitter) * jnp.eye(3)

    Sigma_inv, det = inv_and_det_3x3(Sigma)
    log_det = jnp.log(det)

    # diff.T @ Sigma_inv @ diff
    # Einsum: (N, 3) * (N, 3, 3) * (N, 3) -> (N,)
    # z = Sigma_inv @ diff
    z = jnp.einsum("...ij, ...j -> ...i", Sigma_inv, diff)
    mahalanobis = jnp.sum(diff * z, axis=-1)

    nll = 0.5 * (mahalanobis + log_det)

    return nll


def mass_weighted_hessian_loss(
    label: jnp.array,
    prediction: jnp.array,
    name: str,
    parameters: dict = {},
    inputs: dict = {},
) -> jnp.array:
    """
    Mass-weighted Hessian loss.
    H_mw = H / sqrt(mi * mj)
    """
    label, prediction = label[name], prediction[name]
    Z = inputs["numbers"]

    # Get masses from ase
    masses = jnp.array(atomic_masses)
    m = masses[Z]  # (batch, natoms)

    # Repeat for 3 DOF
    m_dof = jnp.repeat(m, 3, axis=1)  # (batch, 3*natoms)

    # Create mass matrix: sqrt(mi * mj)
    m_matrix = jnp.sqrt(m_dof[:, :, None] * m_dof[:, None, :])

    # Flatten prediction/label if they are 4D (batch, natoms, 3, natoms, 3)
    if prediction.ndim == 5:
        batch_size, n_atoms = prediction.shape[0], prediction.shape[1]
        prediction = prediction.reshape(batch_size, 3 * n_atoms, 3 * n_atoms)
    if label.ndim == 5:
        batch_size, n_atoms = label.shape[0], label.shape[1]
        label = label.reshape(batch_size, 3 * n_atoms, 3 * n_atoms)

    label_mw = label / m_matrix
    prediction_mw = prediction / m_matrix

    return (label_mw - prediction_mw) ** 2


# losses that read the ensemble spread (`<name>_uncertainty` / `<name>_ensemble`),
# not just the mean prediction
crps_loss.uses_spread = True
crps_ensemble_loss.uses_spread = True
nll_loss.uses_spread = True
nll_3x3.uses_spread = True

loss_functions = {
    "mse": weighted_squared_error,
    "huber": weighted_huber_loss,
    "pairwise_mse": pairwise_squared_error,
    "cosine_sim": force_angle_loss,
    "cosine_sim_div_magnitude": force_angle_div_force_label,
    "cosine_sim_exp_magnitude": force_angle_exponential_weight,
    "tril": stress_tril,
    "crps": crps_loss,
    "crps_ensemble": crps_ensemble_loss,
    "nll": nll_loss,
    "nll_3x3": nll_3x3,
    "mw_hessian": mass_weighted_hessian_loss,
}


@dataclasses.dataclass
class Loss:
    """
    Represents a single weighted loss function that is constructed from a `name`
    and a type of comparison metric.
    """

    name: str
    loss_type: str
    weight: float = 1.0
    atoms_exponent: float = 1.0
    parameters: dict = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self):
        if self.loss_type not in loss_functions.keys():
            raise NotImplementedError(
                f"the loss function '{self.loss_type}' is not known."
            )

        self.loss_fn = loss_functions[self.loss_type]

    def __call__(self, inputs: dict, prediction: dict, label: dict) -> float:
        # TODO we may want to insert an additional `mask` argument for this method

        divisor = inputs["n_atoms"] ** self.atoms_exponent
        batch_losses = self.loss_fn(label, prediction, self.name, self.parameters, inputs)

        axes_to_add = len(batch_losses.shape) - 1
        for _ in range(axes_to_add):
            divisor = divisor[..., None]

        arg = batch_losses / divisor
        loss = self.weight * jnp.sum(jnp.mean(arg, axis=0))
        return loss


@dataclasses.dataclass
class LossCollection:
    loss_list: List[Loss]

    def __call__(self, inputs: dict, predictions: dict, labels: dict) -> float:
        total_loss = 0.0
        for single_loss_fn in self.loss_list:
            loss = single_loss_fn(inputs, predictions, labels)
            total_loss = total_loss + loss

        return total_loss
