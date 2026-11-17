"""US-11 / ARCHITECTURE 4.4: evalkit scoring definitions, checked against sklearn and by hand."""

import ast
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import roc_auc_score

import evalkit
from evalkit import metrics, stats
from evalkit.score import score_predictions


def _rand_proba(rng, n, k):
    p = rng.dirichlet(np.ones(k), size=n)
    return p


def test_version_is_semver():
    parts = evalkit.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


@pytest.mark.parametrize("k", [2, 3, 7])
def test_log_loss_matches_sklearn(k):
    rng = np.random.default_rng(k)
    p = _rand_proba(rng, 200, k)
    y = rng.integers(0, k, 200)
    ours = metrics.log_loss(y, p)
    ref = sk_log_loss(y, p, labels=list(range(k)))
    assert ours == pytest.approx(ref, rel=1e-12)


def test_log_loss_clips_zero_probability_at_1e15():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1, 1])
    # first row puts 0 on the true class -> -log(1e-15) / 2
    assert metrics.log_loss(y, p) == pytest.approx(-np.log(1e-15) / 2, rel=1e-6)


def test_align_proba_reorders_and_pads_missing_classes():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    out = metrics.align_proba(proba, ["b", "a"], ["a", "b", "c"])
    np.testing.assert_allclose(out, [[0.8, 0.2, 0.0], [0.4, 0.6, 0.0]])


def test_align_proba_rejects_unknown_class():
    with pytest.raises(ValueError):
        metrics.align_proba(np.ones((1, 2)) / 2, ["a", "z"], ["a", "b"])


def test_prior_proba_uses_training_frequencies():
    p = metrics.prior_proba(np.array([0, 0, 0, 1]), k=3, n=2)
    np.testing.assert_allclose(p, [[0.75, 0.25, 0.0]] * 2)


def test_skill_formula():
    assert metrics.skill(0.3, 0.6) == pytest.approx(0.5)
    assert metrics.skill(0.6, 0.6) == pytest.approx(0.0)
    assert metrics.skill(1.2, 0.6) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        metrics.skill(0.3, 0.0)


def test_prior_predictor_has_skill_zero():
    rng = np.random.default_rng(0)
    ytr = rng.integers(0, 3, 500)
    yte = rng.integers(0, 3, 100)
    prior = metrics.prior_proba(ytr, 3, len(yte))
    ll = metrics.log_loss(yte, prior)
    assert metrics.skill(ll, ll) == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300), 1)  # many ties
    assert metrics.roc_auc_binary(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_single_class_is_none():
    assert metrics.roc_auc_binary(np.zeros(5, dtype=int), np.random.rand(5)) is None


def test_ece_hand_computed():
    # 4 rows, max-class confidences 0.9,0.9,0.6,0.6 ; correct: T,F,T,T
    p = np.array([[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.4, 0.6]])
    y = np.array([0, 1, 1, 1])
    # bin(0.9): conf .9 acc .5 -> |.4| * 2/4 ; bin(0.6): conf .6 acc 1 -> .4*2/4
    assert metrics.ece(y, p) == pytest.approx(0.4)


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(1)
    q = rng.random(200_000)
    y = (rng.random(q.size) < q).astype(int)
    p = np.column_stack([1 - q, q])
    assert metrics.ece(y, p) < 0.01


def test_ece_uses_15_equal_width_bins():
    assert metrics.ECE_BINS == 15


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(-1, 1, allow_nan=False), min_size=2, max_size=30))
def test_bootstrap_interval_contains_mean(values):
    m, lo, hi = stats.bootstrap_mean_ci(np.array(values), n_resamples=2000)
    assert lo - 1e-12 <= m <= hi + 1e-12
    assert m == pytest.approx(float(np.mean(values)))


def test_bootstrap_is_deterministic_and_10k_by_default():
    v = np.arange(10.0)
    assert stats.bootstrap_mean_ci(v) == stats.bootstrap_mean_ci(v)
    assert stats.DEFAULT_RESAMPLES == 10_000


@pytest.mark.parametrize(
    "diffs,expected",
    [
        ([0.10, 0.12, 0.09, 0.11, 0.10], "win"),
        ([-0.10, -0.12, -0.09, -0.11, -0.10], "loss"),
        ([0.10, -0.12, 0.09, -0.11, 0.01], "tie"),
        ([0.0, 0.0, 0.0, 0.0, 0.0], "tie"),
    ],
)
def test_verdict(diffs, expected):
    assert stats.verdict(np.array(diffs)) == expected


def test_wilson_interval_known_value():
    lo, hi = stats.wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_score_predictions_block():
    rng = np.random.default_rng(0)
    p = _rand_proba(rng, 50, 2)
    y_test = np.array(["a", "b"])[rng.integers(0, 2, 50)]
    y_train = np.array(["a", "b"])[rng.integers(0, 2, 200)]
    s = score_predictions(p, ["a", "b"], y_test, y_train, ["a", "b"])
    assert s["scored_by"] == f"evalkit@{evalkit.__version__}"
    assert set(s) == {"scored_by", "log_loss", "prior_log_loss", "skill", "auc", "ece", "n_test"}
    assert s["n_test"] == 50
    assert s["skill"] == pytest.approx(1 - s["log_loss"] / s["prior_log_loss"])


def test_score_predictions_multiclass_has_no_auc():
    rng = np.random.default_rng(0)
    p = _rand_proba(rng, 30, 3)
    cls = ["a", "b", "c"]
    s = score_predictions(p, cls, np.array(cls)[rng.integers(0, 3, 30)], np.array(cls * 10), cls)
    assert s["auc"] is None


def test_score_predictions_rejects_bad_rows():
    p = np.array([[0.5, 0.6]])
    with pytest.raises(ValueError):
        score_predictions(p, ["a", "b"], np.array(["a"]), np.array(["a", "b"]), ["a", "b"])


def test_evalkit_imports_nothing_from_tab_or_bench():
    root = Path(evalkit.__file__).parent
    for f in root.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for n in names:
                assert n.split(".")[0] not in {"tab", "bench", "report", "sklearn"}, (f, n)
