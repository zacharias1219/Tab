"""US-3 / HF-1: no tuner, calibrator or stacker ever sees an outer test index (spy test)."""

import numpy as np

from tab import Tab
from tab.nested import FitObserver, nested_cross_val, outer_splits, assert_disjoint


def _make(observer):
    return Tab(max_trials=4, seed=0, stacking=True, calibrate=True, _observer=observer)


def test_spy_records_every_stage(binary_frame):
    X, y = binary_frame
    obs = FitObserver()
    _make(obs).fit(X, y)
    stages = {s for s, _ in obs.records}
    assert {"preprocess", "model", "calibrator", "stacker"} <= stages


def test_no_component_sees_outer_test_rows(binary_frame):
    X, y = binary_frame
    X = X.set_index(np.arange(1000, 1000 + len(X)))  # non-trivial index
    y.index = X.index
    results = nested_cross_val(_make, X, y, outer=5, seed=0)
    assert len(results) == 5
    for r in results:
        assert r["observer"].records
        assert_disjoint(r["observer"], r["test_index"])
        assert np.isfinite(r["log_loss"])


def test_outer_splits_partition_rows(binary_frame):
    X, y = binary_frame
    splits = outer_splits(y, 5, seed=0)
    test = np.concatenate([t for _, t in splits])
    assert sorted(test.tolist()) == list(range(len(y)))
