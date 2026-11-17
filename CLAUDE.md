# Tab (3A) — working rules

Specs live in `docs/` (PRD.md, ARCHITECTURE.md, MVP_STATE.md); `prereg/PREREG.md` is the pre-registration.

## Commands
- Install: `uv sync --extra dev` (baselines: `--extra bench`)
- Tests: `uv run python -m pytest -q` (FLAML/AutoGluon tests: `-m bench_extra`)
- Lint / types: `uv run ruff check .` · `uv run mypy --strict evalkit && uv run mypy tab`
- Smoke repro: `uv run python -m bench.cli repro`
- CI-equivalent: `make ci`

## Rules
- Existing tests are never edited unless the user asks. A test change goes in its own commit whose message
  starts with `[test-change]` and says why (`ci/test_lock.py` enforces this).
- A bug starts as a failing test, then gets fixed.
- `tab/` never imports flaml, autogluon, evalkit or bench (NFR-6). `bench/systems/` never computes a metric (HF-3).
- Only evalkit produces scores. Systems write `proba.npy` + `meta.json`; the runner measures time and memory.
- Never edit `prereg/PREREG.md` except by appending a dated amendment. Never edit `bench/selected_datasets.json` after freeze (HF-2).
- No `shell=True`, no unpickling run artifacts (`np.load(..., allow_pickle=False)` only).
- Update `docs/MVP_STATE.md` after each phase.

## Gotchas
- Tab is deterministic only in trial mode (`max_trials`); time mode (`ceiling_s`) is for the benchmark.
- OpenML is reached only by `bench/datasets.py`; CI uses the bundled fixtures in `bench/fixtures/`.
