import functools
from typing import Any

import e3x.nn
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from apax.layers.initializers import uniform_range
from apax.utils.convert import str_to_dtype


class GaussianBasis(nn.Module):
    n_basis: int = 7
    r_min: float = 0.5
    r_max: float = 6.0
    dtype: Any = jnp.float32

    def setup(self):
        dtype = str_to_dtype(self.dtype)

        self.betta = self.n_basis**2 / self.r_max**2
        self.rad_norm = (2.0 * self.betta / np.pi) ** 0.25
        shifts = self.r_min + (self.r_max - self.r_min) / self.n_basis * np.arange(
            self.n_basis
        )

        # shape: 1 x n_basis
        shifts = einops.repeat(shifts, "n_basis -> 1 n_basis")
        self.shifts = jnp.asarray(shifts, dtype=dtype)

    def __call__(self, dr):
        dr = einops.repeat(dr, "neighbors -> neighbors 1")
        # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
        distances = self.shifts - dr

        # shape: neighbors x n_basis
        basis = jnp.exp(-self.betta * (distances**2))
        basis = self.rad_norm * basis

        return basis


class BesselBasis(nn.Module):
    """Non-orthogonalized basis functions of Kocer
    https://doi.org/10.1063/1.5086167
    """

    n_basis: int = 7
    r_max: float = 6.0
    dtype: Any = jnp.float32

    def setup(self):
        dtype = str_to_dtype(self.dtype)
        self.n = jnp.arange(self.n_basis, dtype=dtype)

    def __call__(self, dr):
        dr = einops.repeat(dr, "neighbors -> neighbors 1")
        a = (-1) ** self.n * (jnp.sqrt(2) * np.pi / (self.r_max ** (3 / 2)))
        b = (self.n + 1) * (self.n + 2) / jnp.sqrt((self.n + 1) ** 2 * (self.n + 2) ** 2)
        s1 = jnp.sinc((self.n + 1) * dr / self.r_max)
        s2 = jnp.sinc((self.n + 2) * dr / self.r_max)
        basis = a * b * (s1 + s2)
        return basis


class ChebyshevBasis(nn.Module):
    """Scalar (L=0) Chebyshev basis assembled via e3x.

    Uses ``e3x.nn.basic_chebyshev`` as the radial function and
    ``e3x.nn.cosine_cutoff`` as the envelope, wrapped by ``e3x.nn.basis``
    with ``max_degree=0``.

    Unlike ``GaussianBasis`` / ``BesselBasis`` (which take scalar distances),
    this class takes displacement vectors and passes them directly to e3x.
    For L=0 only the norm enters, so the direction has no effect on the output.

    Two class-level flags signal this to ``RadialFunction``:

    - ``takes_displacement = True``  — ``__call__`` receives ``dr_vec (n_pairs, 3)``
    - ``has_cutoff = True``          — envelope is already embedded inside e3x;
                                       ``RadialFunction`` must not reapply one.

    Parameters
    ----------
    n_basis : int
        Number of Chebyshev basis functions.
    r_max : float
        Cutoff radius, used for both the cosine envelope and Chebyshev
        normalisation (the ``limit`` argument of ``basic_chebyshev``).
    dtype : dtype
        Output dtype.
    """

    n_basis: int = 16
    r_max: float = 6.0
    dtype: Any = jnp.float32

    # Non-annotated → not Flax dataclass fields; read by RadialFunction.
    takes_displacement = True
    has_cutoff = True

    @nn.compact
    def __call__(self, dr_vec):
        # dr_vec shape: (n_pairs, 3)
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)

        # Output shape for max_degree=0: (n_pairs, 1, 1, n_basis)
        basis = e3x.nn.basis(
            dr_vec,
            max_degree=0,
            num=self.n_basis,
            radial_fn=functools.partial(
                e3x.nn.basic_chebyshev, limit=self.r_max
            ),
            cutoff_fn=functools.partial(
                e3x.nn.cosine_cutoff, cutoff=self.r_max
            ),
        )

        # Squeeze parity and (l=0, m=0) dims → (n_pairs, n_basis)
        return basis[:, 0, 0, :].astype(dtype)


class EmbeddedRadialFunction(nn.Module):
    """Radial function with environment-aware elemental embeddings and Hadamard gating.

    Elemental embeddings are initialised from a learned lookup table and then
    refined by a single round of scalar message passing over the neighbour
    graph, so each atom's embedding reflects its local chemical environment
    rather than being a fixed species label:

        e⁽⁰⁾ = Embed(Z)                              # (n_atoms, 1, 1, n_basis)
        msg   = MessagePass(e⁽⁰⁾, basis, ...)         # scalar MP, max_degree=0
        e⁽¹⁾ = e⁽⁰⁾ + msg                            # residual update

    The refined per-atom embeddings are then looked up per pair and concatenated
    in order (center atom first, neighbor atom second), forming an asymmetric
    pair representation:

        e_pair = concat(e_i, e_j)                     # (n_pairs, 1, 1, 2*n_basis)

    Finally the radial basis features are projected and Hadamard-gated by the
    pair embedding (FiLM / Option A — gate acts in the output ``n_radial``
    space so gradients to the two projections remain loosely coupled):

        basis_proj = Dense(n_radial)(basis)            # geometric compression
        gate       = 1 + Dense(n_radial)(e_pair)       # species modulation
        output     = basis_proj * gate                 # (n_pairs, n_radial)

    The gate Dense is **zero-kernel-initialised** so the first forward pass is
    equivalent to ``basis_proj`` alone, giving a stable training start.

    Parameters
    ----------
    basis_fn : nn.Module
        Displacement-aware radial basis, e.g. ``ChebyshevBasis``.
        Must accept ``dr_vec (n_pairs, 3)`` and return ``(n_pairs, n_basis)``
        with the cutoff already applied.
    n_radial : int
        Number of output channels per atom-pair.  The elemental embedding
        dimension is tied to ``basis_fn.n_basis``.
    n_species : int
        Number of chemical species supported.
    dtype : dtype
        Compute and output dtype.
    """

    basis_fn: nn.Module = ChebyshevBasis()
    n_radial: int = 16
    n_species: int = 119
    dtype: Any = jnp.float32

    @property
    def r_max(self):
        return self.basis_fn.r_max

    @property
    def _n_radial(self):
        return self.n_radial

    @nn.compact
    def __call__(self, dr_vec, Z, neighbor_idxs):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]
        n_atoms = Z.shape[0]
        n_basis = self.basis_fn.n_basis

        # Radial basis features: (n_pairs, n_basis) → unsqueeze to e3x format
        # (n_pairs, 1, 1, n_basis).  Cutoff already applied inside basis_fn.
        basis = self.basis_fn(dr_vec)[:, None, None, :]

        # Initial per-atom elemental embeddings: (n_atoms, 1, 1, n_basis).
        # Embedding dim is tied to n_basis so basis and embeddings live in the
        # same space, which MessagePass can directly combine.
        embed = e3x.nn.Embed(
            num_embeddings=self.n_species, features=n_basis, dtype=dtype
        )
        e = embed(Z)

        # Scalar message passing (max_degree=0): each atom aggregates radial
        # basis-weighted neighbour embeddings, refining its representation to
        # reflect the local chemical environment.
        msg = e3x.nn.MessagePass(
            max_degree=0, include_pseudotensors=False, dtype=dtype
        )(e, basis, dst_idx=idx_i, src_idx=idx_j, num_segments=n_atoms)
        e = e3x.nn.add(e.astype(dtype), (0.01 * msg).astype(dtype))  # (n_atoms, 1, 1, n_basis)

        # Look up refined embeddings per pair.
        e_i = e[idx_i]  # (n_pairs, 1, 1, n_basis)
        e_j = e[idx_j]  # (n_pairs, 1, 1, n_basis)

        # Ordered pair representation: center and neighbor kept separate.
        e_pair = jnp.concatenate(
            [e_i, e_j], axis=-1
        )  # (n_pairs, 1, 1, 2*n_basis)

        # Geometric compression: (n_pairs, 1, 1, n_basis) → (n_pairs, 1, 1, n_radial)
        basis_proj = e3x.nn.Dense(self.n_radial, use_bias=False, dtype=dtype)(basis)

        # Species gate in n_radial space.  Zero kernel init → gate = 1 at init.
        gate = 1.0 + e3x.nn.Dense(
            self.n_radial,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            dtype=dtype,
        )(e_pair)

        # Hadamard gate; squeeze e3x dims → (n_pairs, n_radial).
        return (basis_proj * gate)[:, 0, 0, :].astype(dtype)


class RadialFunction(nn.Module):
    n_radial: int = 5
    basis_fn: nn.Module = GaussianBasis()
    n_species: int = 119
    emb_init: str = "uniform"
    use_embed_norm: bool = True
    one_sided_dist: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        dtype = str_to_dtype(self.dtype)
        self.r_max = self.basis_fn.r_max
        self.embed_norm = jnp.array(1.0 / np.sqrt(self.basis_fn.n_basis), dtype=dtype)
        if self.one_sided_dist:
            lower_bound = 0.0
        else:
            lower_bound = -1.0

        if self.emb_init is not None:
            self._n_radial = self.n_radial
            if self.emb_init == "uniform":
                emb_initializer = uniform_range(lower_bound, 1.0, dtype=dtype)
                self.embeddings = self.param(
                    "atomic_type_embedding",
                    emb_initializer,
                    (
                        self.n_species,
                        self.n_species,
                        self.n_radial,
                        self.basis_fn.n_basis,
                    ),
                    dtype,
                )
            else:
                raise ValueError(
                    "Currently only uniformly initialized embeddings and no embeddings"
                    " are implemented."
                )
        else:
            self._n_radial = self.basis_fn.n_basis

    def __call__(self, dr, Z_i, Z_j):
        dtype = str_to_dtype(self.dtype)
        dr = dr.astype(dtype)
        # basis shape: neighbors x n_basis
        basis = self.basis_fn(dr)

        if self.emb_init is None:
            radial_function = basis
        else:
            # coeffs shape: n_neighbors x n_radialx n_basis
            species_pair_coeffs = self.embeddings[
                Z_j, Z_i, ...
            ]  # reverse convention to match original
            if self.use_embed_norm:
                species_pair_coeffs = self.embed_norm * species_pair_coeffs

            # radial shape: neighbors x n_radial
            radial_function = einops.einsum(
                species_pair_coeffs, basis, "nbrs radial basis, nbrs basis -> nbrs radial"
            )

        # shape: neighbors
        dr_clipped = jnp.clip(dr, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr_clipped / self.r_max) + 1.0)
        cutoff = einops.repeat(cos_cutoff, "neighbors -> neighbors 1")

        radial_function = radial_function * cutoff

        assert radial_function.dtype == dtype

        return radial_function
