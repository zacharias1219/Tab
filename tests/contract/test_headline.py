"""HF-4: the headline verdict rests on skill score only. AUC cannot override a log-loss tie."""

import numpy as np

from evalkit.headline import build_headline, compare_systems


def _per_fold(values):
    return {f"d{i}": np.array(v, dtype=float) for i, v in enumerate(values)}


def test_auc_win_with_log_loss_tie_prints_tie():
    rng = np.random.default_rng(0)
    # Skill differences centred on 0 -> tie; AUC differences strongly positive.
    skill_tab = _per_fold([0.30 + rng.normal(0, 0.02, 5) for _ in range(20)])
    skill_base = _per_fold([0.30 + rng.normal(0, 0.02, 5) for _ in range(20)])
    auc_tab = _per_fold([np.full(5, 0.95) for _ in range(20)])
    auc_base = _per_fold([np.full(5, 0.70) for _ in range(20)])
    comp = compare_systems("flaml", skill_tab, skill_base, auc_a=auc_tab, auc_b=auc_base)
    assert comp["verdict"] == "tie"
    assert comp["secondary"]["auc_mean_diff"] > 0.2
    head = build_headline("tab", [comp], n_datasets=20)
    assert "tie" in head["statement"]
    assert "win" not in head["statement"].lower().replace("wins", "")
    assert head["metric"] == "skill_score_log_loss"


def test_clear_skill_win_is_win():
    skill_tab = _per_fold([[0.5] * 5 for _ in range(20)])
    skill_base = _per_fold([[0.3] * 5 for _ in range(20)])
    comp = compare_systems("hgb_default", skill_tab, skill_base)
    assert comp["verdict"] == "win"
    assert comp["wins"] == 20


def test_small_n_is_not_a_claim():
    comp = compare_systems("logreg", _per_fold([[0.5]]), _per_fold([[0.3]]))
    head = build_headline("tab", [comp], n_datasets=1)
    assert head["is_claim"] is False
    assert "n = 1" in head["statement"]


def test_only_shared_datasets_are_compared():
    a = _per_fold([[0.5] * 5, [0.5] * 5])
    b = {"d0": np.array([0.4] * 5)}
    comp = compare_systems("x", a, b)
    assert comp["n_datasets"] == 1
