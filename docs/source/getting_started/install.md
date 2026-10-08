# Install

Requires Python ≥ 3.11. GPU support is Linux-only (JAX limitation).

| Use case | Command |
|---|---|
| CPU | `pip install apax` |
| NVIDIA GPU (Linux) | `pip install "apax[cuda]"` |
| Latest from GitHub | `pip install "apax @ git+https://github.com/apax-hub/apax.git"` (add `[cuda]` after `apax` for GPU) |

Optional extras (combine with commas, e.g. `"apax[cuda,openmm]"`):

| Extra | Enables |
|---|---|
| `openmm` | {doc}`../guides/openmm` (OpenMM ≥ 8.5) |
| `ipsuite` | zntrack/IPSuite nodes in `apax.nodes` ({doc}`../guides/workflows`) |
| `optuna` | hyperparameter optimisation node |
| `mlflow` | MLflow logging callback |
| `umap` | UMAP plots in batch selection |

## Check it works

```bash
apax --version                                     # apax 0.x.y
python -c "import jax; print(jax.devices())"       # GPU install: [CudaDevice(id=0), ...]
```

If a GPU install prints only `CpuDevice`, JAX did not find CUDA: see the
[JAX install guide](https://docs.jax.dev/en/latest/installation.html) and {doc}`../guides/troubleshooting`.

## Development install

```bash
pip install uv
git clone https://github.com/apax-hub/apax.git && cd apax
uv sync --all-extras --no-extra cuda   # CPU
uv sync --extra cuda                   # GPU
uv run pytest tests/unit_tests         # quick check
```

:::{note}
`import apax` changes global JAX/TF settings: it enables 64-bit JAX (`jax_enable_x64`), sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (overwriting your value) and silences TensorFlow logging.
Importing `apax.train` hides GPUs from TensorFlow (tf.data is used on CPU only).
:::
