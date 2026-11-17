"""HF-1 canary: a nested CV that lets the outer test fold into fitting must be caught."""

import pytest

from tab import Tab
from tab.nested import FitObserver, assert_disjoint, outer_splits
from tests.canary.leaky import leaky_nested_cv


def test_leaky_nested_cv_is_caught(binary_frame):
    X, y = binary_frame
    splits = outer_splits(y, 5, seed=0)
    obs = FitObserver()
    leaky_nested_cv(lambda o: Tab(max_trials=1, _observer=o), X, y, splits[:1], obs)
    with pytest.raises(AssertionError):
        assert_disjoint(obs, X.index[splits[0][1]])
