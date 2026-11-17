"""Stacking on out-of-fold predictions only (US-10)."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def oof_predictions(
    fit_rows: Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]],
    X: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    k: int,
) -> tuple[np.ndarray, list[dict]]:
    """Generic OOF: `fit_rows(train_rows)` returns a predict function. Records provenance."""
    preds = np.full((len(X), k), np.nan)
    records = []
    for tr, va in folds:
        predict = fit_rows(tr)
        preds[va] = predict(X[va])
        records.append({"fit_rows": np.asarray(tr), "pred_rows": np.asarray(va)})
    return preds, records


def check_oof_provenance(records: Sequence[dict]) -> None:
    """Every prediction used by the meta-learner comes from a model that did not see that row."""
    assert records, "no provenance records"
    for r in records:
        fit = np.asarray(r["fit_rows"])
        pred = np.asarray(r["pred_rows"])
        overlap = np.intersect1d(fit, pred)
        assert overlap.size == 0, (
            f"{overlap.size} rows predicted by a model that was fitted on them"
        )


def meta_features(probas: Sequence[np.ndarray]) -> np.ndarray:
    return np.hstack([np.log(np.clip(p, 1e-6, 1.0)) for p in probas])


class Stacker:
    """Multinomial logistic regression on log OOF probabilities of the base models."""

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C = C
        self.seed = seed

    def fit(self, probas: Sequence[np.ndarray], y: np.ndarray, k: int) -> Stacker:
        self.k = k
        Z = meta_features(probas)
        self.model_ = LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(Z, y)
        return self

    def predict_proba(self, probas: Sequence[np.ndarray]) -> np.ndarray:
        p = self.model_.predict_proba(meta_features(probas))
        out = np.full((p.shape[0], self.k), 1e-12)
        out[:, np.asarray(self.model_.classes_, dtype=np.int64)] = p
        return out / out.sum(axis=1, keepdims=True)
