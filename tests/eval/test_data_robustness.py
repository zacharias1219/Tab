"""Eval spec section 9: 10 data robustness cases, plus NFR-5 envelope errors."""

import numpy as np
import pandas as pd
import pytest

from tab import Tab


def _base(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(np.where(X["a"] + rng.normal(scale=0.5, size=n) > 0, "p", "n"))
    return X, y


def _ok(m, X, k):
    p = m.predict_proba(X)
    assert p.shape == (len(X), k)
    assert np.isfinite(p).all()
    assert np.max(np.abs(p.sum(1) - 1)) <= 1e-9
    return p


def test_rare_class_and_unseen_class_in_fold():
    X, y = _base()
    y = y.copy()
    y.iloc[:2] = "rare"  # 2 rows: some inner folds never see it
    m = Tab(max_trials=3, seed=0).fit(X, y)
    _ok(m, X, 3)


def test_constant_column():
    X, y = _base()
    X["const"] = 7.0
    _ok(Tab(max_trials=2).fit(X, y), X, 2)


def test_all_missing_column():
    X, y = _base()
    X["empty"] = np.nan
    _ok(Tab(max_trials=2).fit(X, y), X, 2)


def test_categorical_over_1000_levels():
    X, y = _base(n=3000)
    X["hc"] = [f"lvl{i}" for i in range(3000)]
    m = Tab(max_trials=2).fit(X, y)
    _ok(m, X, 2)


def test_boolean_dtype_with_na():
    X, y = _base()
    X["flag"] = pd.array([True, False, None, True] * 50, dtype="boolean")
    X["plain_bool"] = X["a"] > 0
    _ok(Tab(max_trials=2).fit(X, y), X, 2)


def test_mixed_dtypes_in_one_column():
    X, y = _base()
    X["mixed"] = pd.Series([1, "two", 3.0, None] * 50, dtype=object)
    _ok(Tab(max_trials=2).fit(X, y), X, 2)


def test_duplicate_column_names():
    X, y = _base()
    X = pd.concat([X, X[["a"]]], axis=1)
    assert list(X.columns).count("a") == 2
    _ok(Tab(max_trials=2).fit(X, y), X, 2)


def test_ten_classes_ok_eleven_raise():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500)})
    y10 = pd.Series(pd.qcut(X["a"] + rng.normal(scale=0.3, size=500), 10, labels=False).astype(str))
    _ok(Tab(max_trials=2).fit(X, y10), X, 10)
    y11 = pd.Series(pd.qcut(X["a"], 11, labels=False).astype(str))
    with pytest.raises(ValueError, match="classes"):
        Tab(max_trials=1).fit(X, y11)


def test_extreme_values():
    X, y = _base()
    X.loc[::17, "b"] = 1e30
    X.loc[::23, "a"] = -1e30
    m = Tab(max_trials=4).fit(X, y)
    _ok(m, X, 2)


def test_nan_target_raises():
    X, y = _base()
    y = y.astype(object)
    y.iloc[0] = None
    with pytest.raises(ValueError, match="target"):
        Tab(max_trials=1).fit(X, y)


def test_single_class_target_raises():
    X, _ = _base()
    with pytest.raises(ValueError, match="classes"):
        Tab(max_trials=1).fit(X, pd.Series(["a"] * len(X)))


# NFR-5 envelope
def test_too_many_rows_raises():
    X = pd.DataFrame({"a": np.zeros(50_001)})
    y = pd.Series(np.arange(50_001) % 2)
    with pytest.raises(ValueError, match="50,000 rows"):
        Tab(max_trials=1).fit(X, y)


def test_too_many_columns_raises():
    X = pd.DataFrame(np.zeros((20, 201)))
    y = pd.Series(np.arange(20) % 2)
    with pytest.raises(ValueError, match="200 columns"):
        Tab(max_trials=1).fit(X, y)


def test_at_the_limit_is_accepted_by_validation():
    from tab.api import validate_inputs

    X = pd.DataFrame(np.zeros((50_000, 3)))
    validate_inputs(X, pd.Series(np.arange(50_000) % 2))
