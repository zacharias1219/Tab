"""US-1 and ARCHITECTURE 4.1: the Tab fit/predict contract."""

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_wine

from tab import Tab


def _check_proba(p, n, k):
    assert isinstance(p, np.ndarray)
    assert p.dtype == np.float64
    assert p.shape == (n, k)
    assert np.all(p >= 0)
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) <= 1e-9


def test_binary_fit_predict(binary_frame):
    X, y = binary_frame
    m = Tab(max_trials=3, seed=0).fit(X, y)
    assert list(m.classes_) == ["neg", "pos"]
    _check_proba(m.predict_proba(X.iloc[:50]), 50, 2)


def test_multiclass_column_order_is_classes_order(multi_frame):
    X, y = multi_frame
    m = Tab(max_trials=3, seed=0).fit(X, y)
    assert list(m.classes_) == sorted(y.unique())
    p = m.predict_proba(X)
    _check_proba(p, len(X), 3)
    # columns follow classes_: the predicted column should agree with the label more often than chance
    acc = np.mean(m.classes_[p.argmax(1)] == y.to_numpy())
    assert acc > 0.6


@pytest.mark.parametrize("loader", [load_iris, load_wine])
def test_tiny_real_data(loader):
    X, y = loader(return_X_y=True, as_frame=True)
    m = Tab(max_trials=2, seed=0).fit(X, y)
    _check_proba(m.predict_proba(X), len(X), len(np.unique(y)))


def test_report_wall_clock_includes_refit(binary_frame):
    X, y = binary_frame
    t0 = time.perf_counter()
    m = Tab(max_trials=3, seed=0).fit(X, y)
    outer = time.perf_counter() - t0
    r = m.report_
    assert 0 < r.actual_wall_s <= outer + 1e-6
    assert r.refit_s > 0
    assert r.actual_wall_s >= r.refit_s
    assert r.n_trials >= 1
    assert r.stop_reason in {"max_trials", "plateau", "ceiling"}
    assert r.leaderboard and r.ece_before is not None


def test_exactly_one_of_ceiling_or_trials():
    with pytest.raises(ValueError):
        Tab()
    with pytest.raises(ValueError):
        Tab(ceiling_s=10, max_trials=3)


def test_time_mode_respects_ceiling(binary_frame):
    X, y = binary_frame
    m = Tab(ceiling_s=6.0, seed=0).fit(X, y)
    assert m.report_.actual_wall_s <= 6.0 * 1.10
    assert m.report_.stop_reason in {"ceiling", "plateau"}
    _check_proba(m.predict_proba(X), len(X), 2)


def test_nan_target_raises(binary_frame):
    X, y = binary_frame
    y = y.copy().astype(object)
    y.iloc[3] = np.nan
    with pytest.raises(ValueError, match="target"):
        Tab(max_trials=1).fit(X, y)


def test_predict_before_fit_raises(binary_frame):
    with pytest.raises(RuntimeError):
        Tab(max_trials=1).predict_proba(binary_frame[0])


def test_predict_accepts_reordered_columns(binary_frame):
    X, y = binary_frame
    m = Tab(max_trials=2, seed=0).fit(X, y)
    p1 = m.predict_proba(X)
    p2 = m.predict_proba(X[list(reversed(X.columns))])
    np.testing.assert_array_equal(p1, p2)


def test_predict_missing_column_raises(binary_frame):
    X, y = binary_frame
    m = Tab(max_trials=1, seed=0).fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X.drop(columns=["x2"]))


def test_integer_labels_roundtrip():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200)})
    y = pd.Series((X["a"] > 0).astype(int) * 5 + 1)
    m = Tab(max_trials=1).fit(X, y)
    assert list(m.classes_) == [1, 6]
