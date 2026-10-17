Uncertainty with Shallow Ensembles
==================================

A shallow ensemble shares the whole model except the last layer of the energy
readout, which has one output column per member. Every member predicts its own
energy, and its forces and stress are that energy's derivatives. The ensemble mean
is the prediction; the spread across members (standard deviation with Bessel's
correction) is the uncertainty. The approach and the training protocols below
follow Kellner & Ceriotti (2024) and Schäfer et al., "How to Train a Shallow
Ensemble" (arXiv:2602.15747).

.. code-block:: yaml

    model:
      ensemble:
        kind: shallow
        n_members: 32
        chunk_size: 4   # optional, see "Cost and memory"

Training
--------

A loss on the ensemble mean (``mse``, ``huber``, ...) gives every member the same
gradient, so the members never move apart: the spread stays whatever the
initialisation made it. At least the energy needs a probabilistic loss, which reads
``<name>_uncertainty`` or ``<name>_ensemble``:

.. list-table::
   :header-rows: 1

   * - ``loss_type``
     - Description
     - ``parameters``
   * - ``nll``
     - Gaussian negative log-likelihood per component, with the ensemble std as sigma.
     - ``eps``: smooth sigma floor, sqrt(sigma² + eps²) (default 1e-6)
   * - ``crps``
     - Continuous ranked probability score of that Gaussian. Bounded gradients, more
       robust than ``nll`` early in training.
     - ``eps``
   * - ``nll_3x3``
     - Forces only: Gaussian NLL of each atom's force with the members' 3x3 covariance.
       Rotation invariant and sensitive to the direction of the uncertainty; needs
       enough members to estimate the covariance (at least ~8).
     - ``rel_jitter`` (default 1e-4, relative to tr(Σ)/3), ``abs_jitter`` (default 1e-12)
   * - ``crps_ensemble``
     - Fair CRPS of the members themselves, no Gaussian assumption. Cost O(members²)
       per component.
     -

Models trained with an energy-only probabilistic loss give poorly calibrated force
uncertainties (Schäfer et al.); for calibrated forces use a probabilistic force loss
as well:

.. code-block:: yaml

    loss:
    - name: energy
      loss_type: nll
    - name: forces
      loss_type: nll
      weight: 4.0

Stress can keep a loss on the mean (``tril``).

Cost and memory
---------------

The energy spread is free. Force and stress spreads need the per-member forces, one
backward pass per member, and a probabilistic force loss differentiates through all
of them. Training time per step for a GMNN model on BMIM-BF4 (300 atoms, batch 8,
RX 9070 XT):

.. list-table::
   :header-rows: 1

   * - Training
     - Time per step
   * - Single model
     - 1x
   * - 32 members, losses on the mean, or energy ``nll`` + force ``mse``
     - 1.0x
   * - 32 members, probabilistic force loss (``chunk_size: 4``)
     - 6.4x

Apax computes the per-member forces only where they are needed.
``force_variance: null`` (the default) means: in training, only if a force or stress
loss uses the spread; if only a metric does, a separate validation model computes
them and the training steps stay cheap. At inference (ASE calculator, ``apax eval``)
they are computed unless ``force_variance: false``. In MD the dynamics always use the
gradient of the mean energy; the per-member forces are computed only for sampled
frames when ``forces_uncertainty`` is a requested property or a dynamics check.

``chunk_size`` computes the per-member forces a few members at a time, each chunk
rematerialised in the backward pass. Without it all members are vmapped at once,
which needs memory proportional to the number of members (32 members did not fit in
16 GB in the example above). Small chunks were also the fastest there (4 better than
2, 8 or 16).

Cheaper protocols
^^^^^^^^^^^^^^^^^

* **Two stages** (Schäfer et al.): train with energy ``nll`` + force ``mse`` (single-model
  cost), then fine-tune briefly with energy and force ``nll``.
* **Loss schedule:** the expensive loss only every few steps or epochs:

  .. code-block:: yaml

      loss_schedule:
        every: 4
        unit: steps   # or epochs
        cheap_loss:
        - name: energy
          loss_type: nll
        - name: forces
          loss_type: mse

  Both losses get their own compiled step; ``train_loss`` and ``train_loss_cheap`` are
  logged separately. On the cheap steps the member spread of the forces is not
  trained, while the backbone keeps changing.

From a trained single model
---------------------------

Transfer learning from a single model into a shallow ensemble copies the output
layer to every member and adds Gaussian noise of ``ensemble_init_noise`` times its
RMS (default 0.1), so that the members differ. Any other parameter whose shape
differs between the two models is an error.

.. code-block:: yaml

    model:
      ensemble: {kind: shallow, n_members: 32, chunk_size: 4}
    loss:
    - {name: energy, loss_type: nll}
    - {name: forces, loss_type: nll, weight: 4.0}
    transfer_learning:
      base_model_checkpoint: models/my_single_model
      ensemble_init_noise: 0.1

Fine-tune the whole model, not only the last layer: Schäfer et al. find last-layer
fine-tuning fails for some systems (BMIM).

With the Muon optimizer, the output layer of every readout (including the
ensemble's) is optimised with Adam; orthogonalised updates would couple the members.

Evaluating uncertainties
------------------------

Metrics for any quantity with an ``_uncertainty`` prediction:

.. list-table::
   :header-rows: 1

   * - Reduction
     - Meaning
   * - ``nll``
     - Gaussian NLL per component.
   * - ``rll``
     - Relative log-likelihood in %: 0 for a constant sigma equal to the RMSE, 100 for
       sigma = abs(error). Comparable across datasets.
   * - ``calib_alpha``
     - The factor alpha that minimises the NLL of alpha * sigma; 1 means calibrated,
       above 1 overconfident.
   * - ``rll_cal``
     - RLL after rescaling sigma by alpha.

.. code-block:: yaml

    metrics:
    - name: energy
      reductions: [mae, rll, rll_cal, calib_alpha]
    - name: forces
      reductions: [mae, rll, rll_cal, calib_alpha]

Fit alpha on validation data, never on training data (training-set uncertainties are
overconfident). Components with zero spread and zero error (fixed by symmetry) are
left out of these scores.

Caveats
-------

* With few members the sample std is noisy and biased; NLL-type losses then
  over-penalise. 16-32 members are typical.
* Force uncertainties can be miscalibrated per element even when the overall alpha is
  close to 1; check them per element.
* NLL down-weights high-uncertainty points (gradient of the mean ~ error/sigma²); if
  the mean fit degrades, use ``crps``, or pretrain with ``mse``.
