"""US-6: a feature that alone determines the label ranks first (20 seeds, >= 19)."""

import numpy as np
import pandas as pd

from tab import Tab
from tab.importance import permutation_importance


def _planted(seed, n=300):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    X["cat"] = pd.Categorical(rng.choice(list("abc"), n))
    planted = f"f{seed % 6}"
    y = pd.Series(np.where(X[planted] > 0, "yes", "no"))
    return X, y, planted


def test_planted_feature_ranks_first_in_19_of_20_seeds():
    hits = 0
    for seed in range(20):
        X, y, planted = _planted(seed)
        m = Tab(max_trials=1, seed=seed, stacking=False, calibrate=False).fit(X.iloc[:200], y.iloc[:200])
        imp = permutation_importance(m, X.iloc[200:], y.iloc[200:], n_repeats=3, seed=seed)
        hits += imp.iloc[0]["feature"] == planted
    assert hits >= 19


def test_importance_frame_shape():
    X, y, _ = _planted(0)
    m = Tab(max_trials=1, seed=0).fit(X.iloc[:200], y.iloc[:200])
    imp = permutation_importance(m, X.iloc[200:], y.iloc[200:], n_repeats=2, seed=0)
    assert list(imp.columns) == ["feature", "importance_mean", "importance_std"]
    assert len(imp) == X.shape[1]
    assert imp["importance_mean"].is_monotonic_decreasing
