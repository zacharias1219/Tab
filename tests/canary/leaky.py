"""Deliberately leaky implementations. Each one MUST be caught by the matching check.

These exist only to prove the checks bite (HF-1). They are never imported by tab/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tab.preprocess import Preprocessor


class LeakyPreprocessor(Preprocessor):
    """Fits its statistics on train plus whatever `peek` frame it was handed (validation rows)."""

    def __init__(self, peek: pd.DataFrame, **kw):
        super().__init__(**kw)
        self.peek = peek

    def fit(self, X: pd.DataFrame):
        return super().fit(pd.concat([X, self.peek], axis=0))


def leaky_nested_cv(make_model, X, y, outer_splits, observer):
    """Fits on ALL rows (outer test included) in every outer fold."""
    out = []
    for _, test_idx in outer_splits:
        m = make_model(observer)
        m.fit(X, y)
        out.append(m.predict_proba(X.iloc[test_idx]))
    return out


def leaky_oof(models_fit_fn, X, y, folds):
    """Produces 'OOF' predictions with models fitted on all rows (in-sample)."""
    full = models_fit_fn(np.arange(len(y)))
    preds = full(X)
    records = [{"fit_rows": np.arange(len(y)), "pred_rows": val} for _, val in folds]
    return preds, records
