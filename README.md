# Tab

A small AutoML system for tabular classification whose product is **honest measurement**, not leaderboard rank.
Every number on the results page traces to a check and to files a stranger can regenerate.

- `tab/`: the library. In-fold preprocessing, a boring model zoo (logistic, random forest, histogram gradient boosting, small MLP), random search over 3 inner folds, a plateau budget controller, out-of-fold stacking, temperature calibration with ECE, permutation importance, a leak detector and forward-chaining time splits.
- `bench/`: the harness. Runs every system, Tab included, as one subprocess per (system, dataset, fold), measured from outside.
- `evalkit/`: the only code allowed to produce a score. Standalone package, no imports from `tab/` or `bench/`.
- `contracts/`: JSON Schemas for `results.json`, prediction metadata, system logs and the leaderboard.
- `report/`: builds the static results page from `results.json` and refuses to build if a section has no data.
- `prereg/PREREG.md`: the pre-registration, committed before any benchmark result.
- `docs/`: PRD and eval spec, architecture and contracts, MVP state, runbook.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (`uv python install 3.12`).

```bash
uv sync --extra dev
```

```bash
uv run python -m bench.cli repro
```

The second command is the smoke repro (US-13): 2 bundled datasets, 1 fold, 20 s ceiling, three systems
(class prior, logistic regression, Tab). It checks every score against `ci/smoke_reference.json` and builds
the page in `report/site/smoke/index.html`. It needs no network and takes well under a minute.

## Use the library

```python
from tab import Tab

model = Tab(ceiling_s=60, seed=0, n_jobs=4).fit(X_train, y_train)   # time mode (benchmark)
proba = model.predict_proba(X_test)                                 # (n, k) float64, columns = model.classes_
model.report_.stop_reason, model.report_.actual_wall_s, model.report_.leak_flags
model.write_leaderboard("leaderboard.json")

Tab(max_trials=20, seed=0).fit(X_train, y_train)                    # trial mode: bitwise deterministic (NFR-2)
Tab(ceiling_s=60, time_col="date").fit(X_train, y_train)            # forward-chaining inner folds
```

Exactly one of `ceiling_s` or `max_trials` must be set. Inputs beyond 50,000 rows, 200 columns or 10 classes raise `ValueError`.

## Commands

Every Makefile target is a thin wrapper around a `python -m` command, so everything works on native Windows without make.

| Purpose | Command |
|---|---|
| Tests (unit, contract, canary, eval) | `uv run python -m pytest -q` |
| Baseline wrapper tests (needs `--extra bench`) | `uv run python -m pytest -q -m bench_extra` |
| Lint, types | `uv run ruff check .` · `uv run mypy --strict evalkit` · `uv run mypy tab` |
| Walking skeleton | `uv run python -m bench.cli skeleton` (add `--dataset credit-g` locally) |
| Smoke repro | `uv run python -m bench.cli repro` |
| Freeze the dataset list (once) | `uv run python -m bench.cli freeze` |
| Tune on the development pool (once) | `uv run python -m bench.tune_dev --cpus 8` |
| Full benchmark and studies | `uv run python -m bench.cli repro-full --cpus 8` |
| Recompute every score from `runs/` | `uv run python -m bench.cli verify --results results/full/results.json` |
| Build the page | `uv run python -m report.build --results results/full/results.json` |
| Pre-registration and freeze checks | `uv run python ci/prereg_order.py` · `uv run python ci/dataset_hash.py` |

## Running the real benchmark

See `docs/RUNBOOK.md`. In short:

1. Paste the frozen tier-1 (20), tier-2 (8) and eligible (54) CC18 task ids into `bench/selected_datasets.json`, run `freeze`, and commit.
2. `uv sync --extra dev --extra bench`, then run `bench.tune_dev`, which writes `prereg/tuned_params.json`. Commit it before any result (HF-6).
3. Run `repro-full`. It resumes after interruption, because a run directory either has a valid record or gets redone.
4. Run `verify`, then commit `results/full/` with `git add -f`. CI publishes the page to GitHub Pages from `main`.

## Contracts in one paragraph

A system process gets `--split-dir --budget-s --seed --cpus --out`. The split dir holds `train.parquet` (features plus `__y`) and `test_X.parquet` only.
The system writes `proba.npy` (float64, rows sum to 1, columns in `meta.classes` order) and `meta.json`, which the schema validates.
`meta.json` cannot carry a score, because unknown fields are rejected. The runner records actual seconds from spawn to exit and peak RSS over the process tree.
It kills the run at 1.5 × the ceiling, and evalkit scores the saved probabilities against labels the system never saw.

## License

See `LICENSE`.
