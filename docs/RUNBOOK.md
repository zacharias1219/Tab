# Runbook

## Machines
- Benchmark machine: the Omen 16 (native Windows, 20 logical CPUs, 16.8 GB RAM). Run everything from a terminal in the repo root with `uv run`.
- CI: GitHub Actions, ubuntu plus windows. CI never touches the network for data, because it uses the bundled fixtures.

## One-time setup before the first benchmark result (order matters: HF-2, HF-6)
1. `uv sync --extra dev --extra bench`
2. Edit `bench/selected_datasets.json`: fill `tier1` (20 entries), `tier2` (8) and `eligible` (54 task ids), copied from "3A Evidence: Spike and Dataset Freeze". Each entry is `{"name": "<openml name>", "task_id": <int>}`.
3. `uv run python -m bench.cli freeze`, then commit `bench/selected_datasets.json` and `bench/selected_datasets.sha256`.
4. `uv run python -m bench.cli skeleton --dataset credit-g` (walking skeleton on real data).
5. `uv run python -m bench.tune_dev --cpus 8` (about 12 datasets x 5 folds x 60 s, roughly 1.5 h). Commit `prereg/tuned_params.json`.
6. `uv run python ci/prereg_order.py` must print OK.

## The benchmark
- `uv run python -m bench.cli repro-full --cpus 8` runs tier 1 x 5 folds x 6 systems at 60 s (about 600 runs, roughly 12 h), then the studies: leakage, stacking ablation, controller (tier 2 x 5 folds x 2 arms at 300 s, roughly 7 h) and time-aware.
- It is resumable: rerun the same command after a crash or reboot. Finished runs are skipped, and partial ones are redone.
- To run only some systems, pass `--systems tab,hgb_default,logreg,prior`.
- The memory guard kills a run whose process tree exceeds `--mem-limit-gb` (default 12) and records status `oom`.

## Checking and publishing
1. `uv run python -m bench.cli verify --results results/full/results.json` recomputes every score from `runs/` and re-hashes every `proba.npy`.
2. `git add -f results/full/results.json` and commit. Do not commit `runs/`: archive it separately (zip) and attach it to a GitHub release, so `verify` can be rerun by others.
3. Push to `main`. The `pages` job builds the page from `results/full/results.json` and deploys it to GitHub Pages.

## Rolling back the page
Revert the commit that changed `results/full/results.json` (`git revert <sha>`) and push. The pages job redeploys the previous page. Previous page builds are also kept as CI artifacts (`results-page`).

## Where to look when something fails
- A single run: `runs/<system>/<dataset>/f<k>_s<seed>/stderr.log`, `record.json`, `system_log.json`.
- Overruns (NFR-1): `analyses.nfr1.overruns` in `results.json` and on the page under "Cost and latency".
- Data drift: `.cache/openml/manifest.json` holds the sha256 of every downloaded file. A mismatch raises on load.
- CI: the failing step name maps to a PRD check ID (see the table in `docs/PRD.md` section 10).

## Owner
The author is the incident owner for the benchmark and the page. There is no running service.
