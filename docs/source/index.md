# apax

**Train machine-learned interatomic potentials and run molecular dynamics with them, in JAX.**

apax fits the Gaussian Moment Neural Network (GMNN) and related models to energies and forces,
then uses them through ASE, its own JAX-MD engine or OpenMM. Everything is driven by two YAML files and a CLI.

```bash
pip install apax                       # CPU; GPU: pip install "apax[cuda]"
apax template train                    # writes config.yaml; fill in the <...> fields
apax train config.yaml                 # trains into models/apax/
apax md config.yaml md_config.yaml     # NVT MD with the trained model
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🚀 Quickstart
:link: getting_started/quickstart
:link-type: doc
Data → trained model → MD in about 5 minutes on a laptop CPU. No downloads.
:::

:::{grid-item-card} 🧭 How-to guides
:link: guides/index
:link-type: doc
Task recipes: prepare data, train, evaluate, run MD, uncertainties, fine-tuning, active learning.
:::

:::{grid-item-card} 📖 Reference
:link: reference/index
:link-type: doc
Every CLI command, every config field with defaults, the files apax writes, the Python API.
:::

:::{grid-item-card} 🎓 Tutorials
:link: _tutorials/index
:link-type: doc
Notebooks: training on ethanol, MD, transfer learning, batch selection, vibrations.
:::
::::

## What do you want to do?

| I want to… | Go to |
|---|---|
| install apax (CPU or GPU) | {doc}`getting_started/install` |
| see the whole workflow once | {doc}`getting_started/quickstart` |
| understand what apax does under the hood | {doc}`getting_started/concepts` |
| turn my DFT data into a training set | {doc}`guides/data` |
| train a model / pick settings / resume | {doc}`guides/training` |
| choose a model / translate MACE settings | {doc}`guides/choosing_a_model` |
| measure test errors | {doc}`guides/evaluation` |
| use a model in ASE (relax, MD, NEB, …) | {doc}`guides/ase` |
| run fast MD on the GPU | {doc}`guides/md` |
| get uncertainties / stop bad MD early | {doc}`guides/uncertainty` |
| fine-tune a pretrained model | {doc}`guides/fine_tuning` |
| pick the most informative new structures | {doc}`guides/active_learning` |
| compute stress, Hessians, vibrations | {doc}`guides/derivatives` |
| add repulsion / predict extra properties | {doc}`guides/physics_terms` |
| use apax from OpenMM | {doc}`guides/openmm` |
| build workflows (IPSuite/zntrack, Optuna) | {doc}`guides/workflows` |
| make it faster / fit in GPU memory | {doc}`guides/performance` |
| fix an error | {doc}`guides/troubleshooting` |
| know current limitations (stress, MD quirks) | {doc}`reference/known_issues` |

## Cite

M. R. Schäfer, N. Segreto, F. Zills, C. Holm, J. Kästner,
[*Apax: A Flexible and Performant Framework for the Development of Machine-Learned Interatomic Potentials*](https://doi.org/10.1021/acs.jcim.5c01221),
J. Chem. Inf. Model. **65**, 8066 (2025). GMNN: Zaverkin & Kästner, [JCTC 16, 5410 (2020)](https://doi.org/10.1021/acs.jctc.0c00347);
Zaverkin et al., [JCTC 17, 6658 (2021)](https://doi.org/10.1021/acs.jctc.1c00527).

Questions: [Discord](https://discord.gg/7ncfwhsnm4) · Bugs: [GitHub issues](https://github.com/apax-hub/apax/issues)

```{toctree}
:hidden:
:caption: Getting started

getting_started/install
getting_started/quickstart
getting_started/concepts
```

```{toctree}
:hidden:
:caption: How-to guides

guides/data
guides/training
guides/choosing_a_model
guides/evaluation
guides/ase
guides/md
guides/uncertainty
guides/fine_tuning
guides/active_learning
guides/derivatives
guides/physics_terms
guides/openmm
guides/workflows
guides/performance
guides/troubleshooting
```

```{toctree}
:hidden:
:caption: Tutorials

_tutorials/index
```

```{toctree}
:hidden:
:caption: Reference

reference/cli
reference/train_config
reference/md_config
reference/files
reference/known_issues
modules/index
```
