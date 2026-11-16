"""Study code runs end to end on bundled data (the real studies need the frozen datasets)."""

import numpy as np

from bench import controller_study, leak_study, time_study
from bench.tune_dev import choose_controller, choose_threshold


def test_leak_study_on_fixtures():
    out = leak_study.run(["fixture_binary", "fixture_multiclass"])
    assert out["n_cases"] == 24
    assert out["simple_set_recall"]["n"] == 8
    assert out["simple_set_recall"]["recall"] >= 0.7
    assert 0 <= out["null_fpr"]["false_positives"] <= out["null_fpr"]["n_null_columns"] == 10
    assert "statement" in out


def test_controller_summary_verdicts():
    def rec(system, d, f, secs, skill):
        return {"system": system, "dataset": d, "fold": f, "actual_wall_s": secs, "status": "ok", "stop_reason": "plateau",
                "scores": {"skill": skill}}

    runs = []
    for i, d in enumerate("abcdefgh"):
        for f in range(5):
            runs.append(rec("tab_fixed", d, f, 300.0, 0.5 + 0.01 * i))
            runs.append(rec("tab", d, f, 150.0 + f, 0.5 + 0.01 * i - 0.001))
    out = controller_study.summarize(runs, margin=0.01)
    assert out["verdict"] == "success"
    assert out["compute_saved"]["estimate"] > 0.45
    out2 = controller_study.summarize(runs, margin=0.0001)
    assert out2["verdict"] == "partial"


def test_choose_controller_respects_loss_budget():
    curves = [[(1.0 * 0.99**i, 1.0) for i in range(30)] + [(0.99**29, 1.0)] * 30 for _ in range(4)]
    out = choose_controller(curves)
    assert out["chosen"]["mean_loss_increase"] <= 0.002
    assert out["chosen"]["mean_saved"] > 0.2


def test_choose_threshold():
    assert choose_threshold({0.5: 0.1, 0.7: 0.03, 0.8: 0.01, 0.9: 0.0}) == 0.8


def test_synthetic_drift_has_known_onset():
    X, y = time_study.synthetic_drift(n=500)
    assert X["t"].is_monotonic_increasing
    assert set(y) == {"a", "b"}


def test_time_ordering_check():
    import pandas as pd

    ok = time_study.check_ordering(pd.DataFrame({"date": np.linspace(0, 1, 200)}), "date")
    bad = time_study.check_ordering(pd.DataFrame({"date": np.random.rand(200)}), "date")
    assert ok["usable"] and not bad["usable"]
