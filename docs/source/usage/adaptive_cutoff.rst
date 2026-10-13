Adaptive Cutoffs
================

With an adaptive cutoff, every atom gets its own cutoff radius, chosen such that it sees approximately a fixed number of neighbors.
In dense systems this keeps the atomic environments (and the memory and compute per atom) bounded, while in sparse systems, such as dimers or molecules in vacuum, the cutoff extends to find neighbors.
This makes it possible to train a single model on dense bulk data alongside dimers or other gas-phase data.
The procedure follows PET (the "solver" method of ``metatrain``).

.. code-block:: yaml

    model:
      basis:
        r_max: 6.0          # maximum cutoff, and neighbor list cutoff
      adaptive_cutoff:
        num_neighbors: 16   # target number of neighbors per atom
        cutoff_width_adaptive: 0.5
        cutoff_width: 0.5
        extra_neighbors: 4

How it works
------------

1. The neighbor list is computed at the maximum cutoff ``basis.r_max``.
2. For each atom, the cutoff ``r_i`` is found by solving ``n_i(r_i) = num_neighbors``, where ``n_i(r) = sum_j bump(r_ij; r, cutoff_width_adaptive) + num_neighbors * (r / r_max)**3`` is a smoothly counted number of neighbors. The cubic term guarantees a unique solution, so isolated atoms get cutoffs close to ``r_max``.
3. A pair is kept if ``r_ij`` is at most the pair cutoff ``(r_i + r_j) / 2``. Kept edges are weighted by a smooth taper of width ``cutoff_width`` that reaches zero at the pair cutoff. This taper replaces the descriptor's fixed cutoff function at ``r_max``.
4. The kept edges are packed into a fixed number of slots per atom, ``k_sel``.

The per-atom cutoffs depend on the positions of atoms that are *not* selected, so gradients flow through the cutoffs: forces and stress remain exact derivatives of the energy.
Hessians neglect the curvature of the cutoffs.

Sizing and overflow
-------------------

``k_sel`` is a shape, not a hyperparameter: it is not saved with the model, but sized from the data at hand as the largest selected neighbor count plus ``extra_neighbors``.
During training (and batch evaluation), it covers all structures in the dataset.
The ASE calculator sizes it for the current structure, and the jax-md MD loop for the initial structure.
If a later geometry selects more neighbors than fit, the calculator recomputes with a larger capacity, and MD discards and redoes the affected chunk of steps.
Larger ``extra_neighbors`` trades padding for fewer recompilations.
