"""Permutation importance on held-out data (US-6). Importance = increase in log loss."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tab._metrics import log_loss


def permutation_importance(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, seed: int = 0
) -> pd.DataFrame:
    classes = list(model.classes_)
    index = {c: i for i, c in enumerate(classes)}
    yy = np.array([index.get(v, -1) for v in pd.Series(y).to_numpy()])
    keep = yy >= 0  # rows whose class the model never saw cannot be scored
    Xk = X.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    yk = yy[keep]
    base = log_loss(yk, model.predict_proba(Xk))
    rng = np.random.RandomState(seed)
    rows = []
    for j, col in enumerate(Xk.columns):
        drops = []
        for _ in range(n_repeats):
            Xp = Xk.copy()
            Xp.isetitem(j, Xk.iloc[:, j].to_numpy()[rng.permutation(len(Xk))])
            drops.append(log_loss(yk, model.predict_proba(Xp)) - base)
        rows.append((str(col), float(np.mean(drops)), float(np.std(drops))))
    out = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_std"])
    return out.sort_values(
        ["importance_mean", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
