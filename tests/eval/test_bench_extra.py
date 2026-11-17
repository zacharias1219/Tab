"""FLAML and AutoGluon wrappers end to end on a fixture (run with -m bench_extra)."""

import pytest

from bench import datasets, runner

pytestmark = pytest.mark.bench_extra


@pytest.mark.parametrize("system", ["flaml", "autogluon"])
def test_baseline_wrapper_end_to_end(tmp_path, system):
    ds = datasets.load("fixture_binary")
    sp = datasets.write_split(ds, 0, tmp_path / "s", tmp_path / "l")
    rec = runner.run_one(system, ds.name, None, 0, 0, 30.0, 2, sp, tmp_path / "runs")
    assert rec["status"] == "ok", rec["error"]
    assert rec["scores"]["skill"] > 0.1
