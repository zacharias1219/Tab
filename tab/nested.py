"""Nested cross-validation and the fit observer used by the spy tests (US-3, HF-1)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from tab._metrics import log_loss


class FitObserver:
    """Records the row labels (DataFrame index values) every fitted component saw."""

    def __init__(self) -> None:
        self.records: list[tuple[str, np.ndarray]] = []
        self.provenance: list[dict[str, np.ndarray]] = []

    def record(self, stage: str, index) -> None:
        self.records.append((stage, np.asarray(index).copy()))

    def record_oof(self, fit_rows, pred_rows) -> None:
        self.provenance.append(
            {"fit_rows": np.asarray(fit_rows).copy(), "pred_rows": np.asarray(pred_rows).copy()}
        )


def assert_disjoint(observer: FitObserver, test_index) -> None:
    test = set(np.asarray(test_index).tolist())
    for stage, idx in observer.records:
        leaked = test.intersection(np.asarray(idx).tolist())
        assert not leaked, f"stage {stage!r} was fitted on {len(leaked)} outer-test rows"


def inner_folds(
    y_codes: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = np.bincount(y_codes)
    counts = counts[counts > 0]
    n = y_codes.size
    if counts.min() >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in cv.split(np.zeros(n), y_codes)]
    cv2 = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in cv2.split(np.zeros(n))]


def outer_splits(
    y: pd.Series, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    codes = np.unique(pd.Series(y).astype(str).to_numpy(), return_inverse=True)[1]
    return inner_folds(codes, n_splits, seed)


def nested_cross_val(
    make_model: Callable[[FitObserver], Any],
    X: pd.DataFrame,
    y: pd.Series,
    outer: int = 5,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Outer k-fold around a Tab-like estimator whose own fit runs the inner CV.

    The estimator only ever receives outer-training rows; each fold gets a fresh observer so the
    caller can assert no component saw an outer-test row."""
    out = []
    for i, (tr, te) in enumerate(outer_splits(y, outer, seed)):
        obs = FitObserver()
        model = make_model(obs)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])
        classes = list(model.classes_)
        idx = {c: j for j, c in enumerate(classes)}
        yt = y.iloc[te].to_numpy()
        known = np.array([v in idx for v in yt])
        codes = np.array([idx[v] for v in yt[known]], dtype=np.int64)
        ll = log_loss(codes, p[known]) if known.any() else float("nan")
        out.append(
            {
                "fold": i,
                "test_index": X.index[te],
                "observer": obs,
                "log_loss": ll,
                "n_test": len(te),
            }
        )
    return out
