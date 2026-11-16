"""US-11 / HF-3 / HF-5 / threat model: the benchmark harness contract."""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bench import datasets, runner
from evalkit.schema import validate

ROOT = Path(__file__).resolve().parents[2]

FAKES = {
    "sleeper": "tests.fake_systems.sleeper",
    "crasher": "tests.fake_systems.crasher",
    "self_scorer": "tests.fake_systems.self_scorer",
    "bad_rows": "tests.fake_systems.bad_rows",
}


@pytest.fixture
def work(tmp_path):
    return {"splits": tmp_path / "splits", "labels": tmp_path / "labels", "runs": tmp_path / "runs"}


def _split(work, name="fixture_binary", fold=0):
    ds = datasets.load(name)
    return ds, datasets.write_split(ds, fold, work["splits"], work["labels"])


def _run(work, system, ds, sp, budget=20.0, registry=None, **kw):
    return runner.run_one(
        system=system, dataset=ds.name, task_id=ds.task_id, fold=sp.fold, seed=0, budget_s=budget, cpus=1,
        split=sp, runs_root=work["runs"], registry=registry or runner.SYSTEMS, **kw,
    )


@pytest.mark.parametrize("name", ["fixture_binary", "fixture_multiclass"])
def test_stub_system_end_to_end(work, name):
    ds, sp = _split(work, name)
    rec = _run(work, "prior", ds, sp)
    assert rec["status"] == "ok", rec["error"]
    assert rec["scores"]["scored_by"].startswith("evalkit@")
    assert rec["scores"]["skill"] == pytest.approx(0.0, abs=1e-12)  # prior stub has skill 0 by definition
    assert rec["actual_wall_s"] > 0 and rec["peak_rss_gb_tree"] > 0
    rundir = Path(work["runs"]) / "prior" / name / "f0_s0"
    assert (rundir / "proba.npy").exists() and (rundir / "meta.json").exists()
    validate(json.loads((rundir / "meta.json").read_text()), "predictions_meta")


def test_run_record_is_valid_inside_results(work, valid_results):
    ds, sp = _split(work)
    rec = _run(work, "logreg", ds, sp)
    assert rec["status"] == "ok"
    valid_results["runs"] = [rec]
    validate(valid_results, "results")


def test_split_dir_holds_no_labels_and_cli_has_no_label_path(work):
    ds, sp = _split(work)
    names = {p.name for p in Path(sp.split_dir).iterdir()}
    assert names == {"train.parquet", "test_X.parquet"}
    import pandas as pd

    assert "__y" not in pd.read_parquet(Path(sp.split_dir) / "test_X.parquet").columns
    args = runner.build_command("prior", runner.SYSTEMS, sp.split_dir, 20, 0, 1, Path(work["runs"]) / "x")
    joined = " ".join(map(str, args))
    # (the pytest tmp dir name itself contains "labels", so check the label paths, not the word)
    assert str(sp.label_dir) not in joined and "test_y" not in joined and "classes.json" not in joined
    assert Path(sp.label_dir).resolve() not in [Path(sp.split_dir).resolve(), *Path(sp.split_dir).resolve().parents]


def test_timeout_is_killed_and_kept(work):
    ds, sp = _split(work)
    rec = _run(work, "sleeper", ds, sp, budget=1.0, registry={**runner.SYSTEMS, **FAKES})
    assert rec["status"] == "timeout"
    assert rec["error"] and rec["scores"] is None
    assert 1.4 <= rec["actual_wall_s"] < 10


def test_crash_is_recorded_with_error(work):
    ds, sp = _split(work)
    rec = _run(work, "crasher", ds, sp, registry={**runner.SYSTEMS, **FAKES})
    assert rec["status"] == "error"
    assert "deliberate crash" in rec["error"]


def test_system_cannot_report_its_own_score(work):  # HF-3
    ds, sp = _split(work)
    rec = _run(work, "self_scorer", ds, sp, registry={**runner.SYSTEMS, **FAKES})
    assert rec["status"] == "error"
    assert "meta" in rec["error"]


def test_invalid_probabilities_are_an_error(work):
    ds, sp = _split(work)
    rec = _run(work, "bad_rows", ds, sp, registry={**runner.SYSTEMS, **FAKES})
    assert rec["status"] == "error"
    assert "sum" in rec["error"]


def test_resume_reuses_valid_record(work):
    ds, sp = _split(work)
    a = _run(work, "prior", ds, sp)
    b = _run(work, "prior", ds, sp)
    assert a == b


def test_unknown_or_malicious_system_name_rejected(work):
    ds, sp = _split(work)
    with pytest.raises(ValueError):
        _run(work, "prior; rm -rf /", ds, sp)
    with pytest.raises(ValueError):
        _run(work, "not_registered", ds, sp)


def test_run_dir_must_stay_under_runs_root(work):
    with pytest.raises(ValueError):
        runner.run_dir(work["runs"], "prior", "../../etc", 0, 0)


def test_systems_never_compute_metrics():  # HF-3
    banned_mods = {"evalkit", "sklearn.metrics"}
    banned_calls = {"log_loss", "roc_auc_score", "accuracy_score", "brier_score_loss", "score_predictions"}
    for f in (ROOT / "bench" / "systems").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(node.module == m or node.module.startswith(m + ".") for m in banned_mods), f
                assert not {a.name for a in node.names} & banned_calls, f
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not any(a.name == m or a.name.startswith(m + ".") for m in banned_mods), f
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                assert name not in banned_calls, f"{f}: calls {name}"


def test_no_shell_true_anywhere_in_bench():
    for f in (ROOT / "bench").rglob("*.py"):
        assert "shell=True" not in f.read_text(encoding="utf-8"), f


def test_proba_loaded_without_pickle():
    src = (ROOT / "bench" / "runner.py").read_text(encoding="utf-8")
    assert "allow_pickle=False" in src
    assert "pickle" not in src.replace("allow_pickle=False", "")
