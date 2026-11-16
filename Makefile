# Thin wrappers. Every target is also a plain `python -m ...` command (see README) for native Windows.
PY ?= uv run python

.PHONY: install lint typecheck test test-fast skeleton repro repro-full verify page freeze tune-dev ci prereg-check

install:
	uv sync --extra dev

install-bench:
	uv sync --extra dev --extra bench

lint:
	uv run ruff check . && uv run ruff format --check tab evalkit bench report ci

typecheck:
	uv run mypy --strict evalkit && uv run mypy tab

test:
	$(PY) ci/timing_gate.py 180 -- $(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -x tests/unit tests/contract

skeleton:
	$(PY) -m bench.cli skeleton

repro:
	$(PY) -m bench.cli repro

repro-full:
	$(PY) -m bench.cli repro-full --cpus 8

verify:
	$(PY) -m bench.cli verify --results results/full/results.json

page:
	$(PY) -m report.build --results results/full/results.json

freeze:
	$(PY) -m bench.cli freeze

tune-dev:
	$(PY) -m bench.tune_dev --cpus 8

prereg-check:
	$(PY) ci/prereg_order.py && $(PY) ci/dataset_hash.py

ci: lint typecheck test prereg-check repro
