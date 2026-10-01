"""Padded-row neighbour layout: pair p of centre n lands in cell (n, k), k < K.

Moments become reductions over the minor K axis instead of segment sums, so
neither the forward nor the force backward pass contains an atomic scatter.
"""

import jax.numpy as jnp
import numpy as np

# independent components of symmetric rank-2 / rank-3 tensors and their multiplicities
P2 = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
P3 = [
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2),
    (0, 2, 2), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2),
]  # fmt: skip
W2 = np.array([1, 2, 2, 1, 2, 1])
W3 = np.array([1, 3, 3, 3, 6, 3, 1, 3, 3, 1])
UNPACK2 = [P2.index(tuple(sorted(ab))) for ab in np.ndindex(3, 3)]
UNPACK3 = [P3.index(tuple(sorted(abc))) for abc in np.ndindex(3, 3, 3)]


def row_cells(centre, valid, n_atoms, K):
    """Flat cell n * K + k of each pair; invalid or overflowing pairs get unique
    out-of-bounds cells so every scatter into rows has unique indices."""
    P = centre.shape[0]
    key = jnp.where(valid, centre, n_atoms)
    order = jnp.argsort(key, stable=True)
    key_s = key[order]
    k_s = jnp.arange(P) - jnp.searchsorted(key_s, key_s, side="left")
    oob = n_atoms * K + jnp.arange(P)
    cell_s = jnp.where((key_s < n_atoms) & (k_s < K), key_s * K + k_s, oob)
    return jnp.zeros(P, jnp.int32).at[order].set(cell_s, unique_indices=True)


def to_rows(x, cell, n_atoms, K):
    """(P, c) -> (c, n_atoms, K), empty cells zero. Scatter-set with unique
    indices: its transpose is a gather, and that gather's transpose is again a
    non-atomic scatter."""
    rows = jnp.zeros((x.shape[1], n_atoms * K), x.dtype)
    rows = rows.at[:, cell].set(x.T, mode="drop", unique_indices=True)
    return rows.reshape(x.shape[1], n_atoms, K)


def unpack(m, idx):
    return jnp.stack([m[..., i] for i in idx], axis=-1)


def row_moments(f, d):
    """f: (r, N, K), d: (3, N, K). Returns m0 (N, r), m1 (N, r, 3) and the packed
    symmetric m2 (N, r, 6), m3 (N, r, 10). Broadcast-multiply + sum over the minor
    axis: XLA fuses the products into one reduction per order, no einsum/TF32."""
    x2 = jnp.stack([d[a] * d[b] for a, b in P2])
    x3 = jnp.stack([d[a] * d[b] * d[c] for a, b, c in P3])

    def outer(a, b):
        return (a[:, None] * b[None]).sum(-1).transpose(2, 0, 1)

    return [f.sum(-1).T, outer(f, d), outer(f, x2), outer(f, x3)]
