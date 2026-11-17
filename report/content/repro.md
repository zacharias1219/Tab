Smoke repro (2 datasets, 1 fold, 20 s ceiling, three systems; under 15 minutes, no network needed):

    git clone <this repo> && cd tab
    uv sync --extra dev
    uv run python -m bench.cli repro

The command checks every smoke score against the committed reference intervals in `ci/smoke_reference.json` and exits non-zero on any miss. `python -m bench.cli repro-full` regenerates everything on the frozen datasets (needs network for OpenML and the `bench` extra). `python -m bench.cli verify` recomputes every score from `runs/` and compares it with `results.json`.
