GPU Performance
===============

Measured on an AMD RX 9070 XT (ROCm under WSL2) for a GMNN model on BMIM-BF4
(2400 atoms in MD; 300-atom structures, batch 8, in training):

.. list-table::
   :header-rows: 1

   * - Workload
     - Before
     - After
     - Speedup
   * - NPT MD (steps/s)
     - 22.1
     - 161–164
     - 7.3x
   * - NVT MD (steps/s)
     - 53.6
     - 191.5
     - 3.6x
   * - NVE MD (µs/atom/step)
     - 6.11
     - 2.40
     - 2.5x
   * - Training step (ms)
     - 209
     - 30.6
     - 6.8x

Most of the gain comes from avoiding contended floating-point atomics (scatter-adds
of many pairs into few atoms), which are particularly slow on AMD GPUs, and from
fewer, larger kernels in the integrators.

What changed
------------

* **Sorted neighbour lists.** Every neighbour list fed to the model is sorted by
  centre atom (``neighbor_idxs[1]``), padding included, and the descriptor's segment
  sums and centre-atom gather are flagged ``indices_are_sorted=True``. XLA then
  accumulates per segment instead of with atomics.
* **Second derivatives keep the flag.** JAX's transpose of ``scatter_add`` drops
  ``indices_are_sorted`` (it matters for training, which differentiates the forces);
  ``apax.utils.jax_patches`` restores it on import.
* **Species embedding gradient as a matrix product.** The radial function's
  per-species embeddings are looked up with a ``custom_jvp`` whose tangent is a one-hot
  matrix product, instead of scatter-adding every pair into a handful of rows.
* **f32 pair geometry in MD.** ``geometry_dtype`` (models, builder) sets the precision
  of positions, distances and the neighbour list. jax-md MD uses f32 (forces change by
  ~1e-4 eV/Å); the integrator stays f64. Training and the ASE calculator stay f64.
* **Integrators.** NVT and NPT use a Nosé–Hoover chain unrolled into a few kernels
  (identical to jax-md's to 1e-15), and NPT carries dU/dV in its state, so each step
  evaluates the model once instead of twice. Requires jax-md >= 0.2.29.

Recommended setting on ROCm
---------------------------

On ROCm (measured under WSL2), XLA's command buffers (HIP graphs) made MD 1.4–1.7x
slower. Disable them:

.. code-block:: bash

    export XLA_FLAGS=--xla_gpu_enable_command_buffer=

For contributors
----------------

``indices_are_sorted=True`` is an unchecked promise to XLA: an unsorted neighbour list
gives wrong results, not an error. Any new source of neighbour lists must sort by
centre atom and pad with ``apax.data.preprocessing.sort_by_centre`` and
``pad_sorted_nl`` (jax-md's sparse neighbour list is sorted by construction).
