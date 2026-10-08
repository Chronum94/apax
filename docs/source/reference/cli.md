# CLI

`apax <command> --help` shows the options of any command.

| Command | Does |
|---|---|
| `apax train CONFIG` | train (resumes if the model directory has a `latest/` checkpoint). `--log-level debug\|info\|warning` |
| `apax eval CONFIG` | test-set metrics with the `best/` checkpoint → `<model dir>/eval/log.csv`. `--n-data N` |
| `apax md TRAIN_CONFIG MD_CONFIG` | JAX-MD simulation with the model of TRAIN_CONFIG |
| `apax template train [--full]` | write `config.yaml` (minimal) or `config_full.yaml` (all options); never overwrites |
| `apax template md` | write `md_config.yaml` |
| `apax validate train CONFIG [--template]` | check a training config (`--template`: allow missing data paths, for workflow templates) |
| `apax validate md CONFIG` | check an MD config |
| `apax schema train [SECTION] [--keywords] [--flat]` | print the JSON schema; `SECTION` like `model.basis`; `--keywords` lists sub-keys; `--flat` prints a key/type/default table |
| `apax schema md [SECTION] …` | same for MD configs |
| `apax schema vscode` | write JSON schemas to `.vscode/` and register them for `train*.yaml` / `md*.yaml` (autocomplete + linting in VS Code with the YAML extension) |
| `apax visualize CONFIG` | print the model's layer/parameter table |
| `apax docs` | open this documentation |
| `apax --version` | print the version |

Quick key lookup without opening the docs:

```bash
apax schema train model.gmnn --keywords # which keys does a gmnn model accept?
apax schema train --flat | grep lr       # all learning-rate keys with defaults
```
