# Brief for invariant-test agents (delete before merging)

Goal: plug holes in apax with tests that must hold for ANY weights and ANY configuration (tautology / reference
implementation tests). If such a test fails, the code is wrong.

## Setup
- Worktree (branch `invariant-tests`, based on `model-improvements`):
  /tmp/claude-1000/-home-chronum-apax/9a2463cb-3240-48e0-b03c-55467fce4db9/scratchpad/apax-inv
- Run: `cd <worktree> && PYTHONPATH=<worktree> taskset -c <your cores> ~/miniconda3/envs/apaxenv/bin/python -m pytest -q tests/invariants/<your file> -p no:cacheprovider -W ignore`
- Shared helpers: `tests/invariants/conftest.py` (random clusters/bulk cells of many shapes, fp64 random-weight models
  scaled to O(1) outputs, training-path `offset_path`, real `ASECalculator` via `ase_calculator`, `fd_forces`,
  `fd_stress`, hypothesis profile "apax"). Import helpers with `from conftest import ...` is NOT possible across
  rootdirs — use `from tests.invariants.conftest import ...` or rely on the module being importable as
  `conftest` from within the directory (pytest rootdir puts it on sys.path; check what works and be consistent).
- Hypothesis is installed: use `@given` with small strategies (seeds, shapes, species) where it adds coverage.

## Rules
- FIRST invoke the Skill tool with skill `ponytail:ponytail` (full) and follow it: minimal, boring, readable tests.
- Write ONLY your own file(s). Do NOT edit apax/ source, conftest.py or other agents' files. If you need a new shared
  helper, put it at the top of your own file and list it in your report as a conftest candidate.
- fp64 everywhere; tight tolerances (1e-8..1e-10 relative to the scale of the quantity) unless physics dictates.
- Keep your file's runtime ≤ ~3 min on 2 cores. Small systems (≤ ~100 atoms), few hypothesis examples.
- When a test fails: decide carefully whether it is a test bug or an apax bug. Reproduce independently (second
  method) before calling it an apax bug. For confirmed apax bugs: `pytest.mark.xfail(strict=True, reason="<one-line
  user-facing description of the bug>")` on the minimal failing case, and keep a passing variant if one exists.
- After writing: self-review your file (redundancy, weak assertions, tests that can't fail, slow parts) and iterate.

## Report (final message, ≤ 250 words)
Tests added (count, what they cover), runtime, apax bugs found (each: test id, symptom, evidence), conftest
candidates, anything you could not test and why.
