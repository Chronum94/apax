Tutorials
=========

Executable notebooks (in ``examples/`` of the repository). They download small public datasets
(ethanol, benzene) and run on a CPU in minutes.

.. list-table::
   :header-rows: 1

   * - Notebook
     - Teaches
   * - 01 Model training
     - template → validate → train → evaluate, plotting learning curves (ethanol CCSD(T))
   * - 02 Molecular dynamics
     - ASE MD, ``apax md`` with JAX-MD, OpenMM
   * - 03 Transfer learning
     - pretrain on DFT, fine-tune on CCSD(T) (benzene)
   * - 04 Batch data selection
     - pick new training structures from an MD pool
   * - 05 Vibrational analysis
     - Hessians and frequencies

.. note::
   Known issues in the current notebooks: in 03 use ``transfer_learning:`` instead of ``checkpoints:``
   (and ``emb_lr`` instead of ``embedding_lr``); in 05 create the calculator with ``ASECalculator(model_dir, calc_hessian=True)``;
   when starting from ``apax template train --full``, delete the ``transfer_learning`` section unless fine-tuning.

.. toctree::
   :maxdepth: 1

   01_Model_Training
   02_Molecular_dynamics
   03_Transfer_Learning
   04_Batch_Data_Selection
   05_Vibrational_Analysis
