"""US-9: forward-chaining splits. Every validation timestamp is at or after the latest training
timestamp plus the gap. Property test plus the 6 pre-specified edge cases."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tab.time_split import forward_chaining_splits


def _assert_forward(ts, splits, gap=0):
    ts = np.asarray(ts)
    for tr, va in splits:
        assert len(tr) and len(va)
        assert not set(tr) & set(va)
        assert ts[va].min() >= ts[tr].max() + gap


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(0, 50), min_size=30, max_size=200),
    st.integers(2, 4),
    st.integers(0, 3),
)
def test_property_forward_order(ts, n_splits, gap):
    ts = np.array(ts)
    try:
        splits = forward_chaining_splits(ts, n_splits=n_splits, gap=gap)
    except ValueError:
        return  # too few distinct timestamps for the request: a clear error is acceptable
    assert len(splits) == n_splits
    _assert_forward(ts, splits, gap)


def test_tied_timestamps_never_straddle_a_boundary():
    ts = np.repeat(np.arange(10), 10)
    splits = forward_chaining_splits(ts, n_splits=3)
    _assert_forward(ts, splits)
    for tr, va in splits:
        assert ts[va].min() > ts[tr].max()


def test_non_monotone_index_returns_original_positions():
    rng = np.random.default_rng(0)
    ts = rng.permutation(100)
    splits = forward_chaining_splits(ts, n_splits=4)
    _assert_forward(ts, splits)
    assert all(np.all(va < 100) for _, va in splits)


def test_missing_timestamps_raise():
    ts = np.array([1.0, 2.0, np.nan, 4.0] * 10)
    with pytest.raises(ValueError, match="missing"):
        forward_chaining_splits(ts, n_splits=2)


def test_gap_larger_than_data_raises():
    ts = np.arange(20)
    with pytest.raises(ValueError, match="gap"):
        forward_chaining_splits(ts, n_splits=2, gap=100)


def test_single_class_validation_fold_warns():
    ts = np.arange(60)
    y = np.array([0] * 30 + [1] * 15 + [0] * 15)
    with pytest.warns(UserWarning, match="single class"):
        splits = forward_chaining_splits(ts, n_splits=3, y=y)
    _assert_forward(ts, splits)


def test_duplicate_timestamps_with_datetimes():
    ts = pd.to_datetime(["2020-01-01"] * 5 + ["2020-01-02"] * 5 + ["2020-01-03"] * 5 + ["2020-01-04"] * 5)
    splits = forward_chaining_splits(ts, n_splits=3)
    _assert_forward(ts.values, splits)


def test_gap_in_time_units():
    ts = np.arange(100)
    splits = forward_chaining_splits(ts, n_splits=3, gap=5)
    _assert_forward(ts, splits, gap=5)
