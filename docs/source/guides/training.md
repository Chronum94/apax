# Train a model

```bash
apax template train            # minimal config.yaml  (--full: every option, see warning below)
apax validate train config.yaml
apax train config.yaml         # --log-level debug for more output
```

From Python: `from apax.train.run import run; run("config.yaml")` (also accepts a dict).

## Settings that matter

| Setting | Default | Advice |
|---|---|---|
| `n_epochs` | required | 100–2000; watch `val_loss` in `log.csv` |
| `data.batch_size` | **1** | always set: 4–32 for molecules/small cells, 1–4 for large cells |
| `data.valid_batch_size` | 10 | as large as memory allows (speed only) |
| `model.basis.r_max` | 5.0 Å | 5–6 Å; cost grows ~r³ |
| `model.nn` | `[256, 256]` | `[128, 128]` or `[64, 64]`: faster, often equally accurate on small data |
| `model.n_radial` | 5 | 3–5 |
| `forces` loss `weight` | 1.0 | raise (4–10) to favour forces over energies |
| `optimizer.schedule` | `linear` decay to 1e-6 | `cyclic_cosine` (`period`, `decay_factor`) for long runs |
| `patience` | off | e.g. 100 to stop when `val_loss` stalls |
| `weight_average` | off | EMA of weights for validation (see below) |

Any `model:` section must contain `name: gmnn` (the model type; see {doc}`choosing_a_model`).
A good first config: the minimal template plus `batch_size: 8`, `forces` weight 4, 500–1000 epochs, and
`ckpt_interval: 10` so interrupted runs can resume.

## Loss

```yaml
loss:
  - name: energy
  - name: forces
    weight: 4.0
```

Exact definition, per structure with *N* atoms, averaged over the batch and summed over terms:

| Term | Contribution (`loss_type: mse`) |
|---|---|
| energy | weight · (E_pred − E_ref)² / N^`atoms_exponent` |
| forces | weight · Σ_atoms,xyz (F_pred − F_ref)² / N^`atoms_exponent` |
| stress (`tril`) | weight · Σ_lower triangle (σV_pred − σV_ref)² / N^`atoms_exponent` (σV = stress × volume, eV) |

`atoms_exponent` defaults to 1; `atoms_exponent: 2` on the energy term gives the squared per-atom energy error
(MACE convention). Other `loss_type`s: `huber` (`parameters: {delta: 0.1}`), `cosine_sim*` (force direction),
`nll`/`crps` (shallow ensembles, {doc}`uncertainty`), `mw_hessian` ({doc}`derivatives`).
Stress training: read {doc}`../reference/known_issues` first.

## Monitor

`<model dir>/log.csv`, one row per epoch: `train_loss`, `val_loss`, and each `metrics` entry as
`{train,val}_{name}_{reduction}` (e.g. `val_forces_mae`). Energy `mae` is **per structure** (eV); use `per_atom_mae`
for per atom. TensorBoard: `callbacks: [{name: csv}, {name: tensorboard}]`. MLflow:
`{name: mlflow, experiment: my_exp}` (needs `apax[mlflow]`).

## Resume and checkpoints

Re-running `apax train config.yaml` resumes from `latest/`, or exits immediately if training already finished.
`latest/` is written every `ckpt_interval` epochs (default **500**): set a small value for resumable runs.
`best/` is updated whenever `val_loss` improves; all inference tools load `best/`.

## EMA (weight averaging)

`weight_average: {ema_start: 0, alpha: 0.1}`: ema ← (1 − α)·ema + α·params, updated twice per epoch (not per step).
`alpha` is the weight of the *new* parameters. EMA parameters are used for validation, so `best/`, `patience` and
`val_loss` follow the EMA; the saved checkpoints hold the raw parameters. Small `alpha` makes validation lag behind
training; MACE's per-step `ema_decay` has no direct equivalent — start with `alpha` 0.1–0.3.

## Multiple GPUs

`data_parallel: true` (default) splits each batch across all visible GPUs; use a `batch_size` that is a multiple of
the GPU count. Restrict with `CUDA_VISIBLE_DEVICES=0`. (`apax md` uses one GPU.)

## Ensembles

```yaml
model:
  name: gmnn
  ensemble:
    kind: shallow       # n readout heads on a shared descriptor
    n_members: 8
```

`kind: full` trains n independent models in parallel (n× cost). Costs and uncertainty use: {doc}`uncertainty`.

:::{warning}
`apax template train --full` writes `transfer_learning: {base_model_checkpoint: null, ...}`. A `transfer_learning`
section enables transfer learning even with a null checkpoint, and training crashes with
`TypeError: argument should be a str or an os.PathLike object`. Delete that section unless fine-tuning ({doc}`fine_tuning`).
:::

Every option: {doc}`../reference/train_config`. MACE users: {doc}`choosing_a_model` has a key translation table.
