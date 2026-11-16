"""Leak injection for the eval cases and the 240-case study (PRD section 9, PREREG section 7)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TYPES = ["T1", "T2", "T3", "T4"]
SEVERITIES: dict[str, list[str]] = {
    "T1": ["exact", "affine", "label_encoded"],
    "T2": ["sigma0.1", "sigma0.5", "sigma1.0"],
    "T3": ["fully_sorted", "sorted80_blocks", "shuffled_blocks50"],
    "T4": ["1%", "5%", "20%"],
}
FLIP_SEVERITIES = ["flip0.05", "flip0.2", "flip0.4"]
SIMPLE_SET = {("T1", "exact"), ("T1", "affine"), ("T3", "fully_sorted"), ("T4", "20%")}
LEAK_COL = "__leak__"


def _codes(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(y.astype(str).to_numpy(), return_inverse=True)
    return inv.astype(np.int64), uniq


def inject(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    leak_type: str,
    severity: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    """Return (X_train', y_train', X_test', expected). Test labels are used only for T4, where
    duplicated test rows carry their own labels into train, exactly as a real overlap would."""
    rng = np.random.RandomState(seed)
    Xtr, ytr, Xte = X_train.copy(), y_train.copy(), X_test.copy()
    codes, _ = _codes(pd.concat([ytr, y_test]))
    ctr = codes[: len(ytr)]
    cte = codes[len(ytr) :]

    if leak_type == "T1":
        if severity == "exact":
            Xtr[LEAK_COL], Xte[LEAK_COL] = ytr.astype(str).to_numpy(), y_test.astype(str).to_numpy()
            Xtr[LEAK_COL] = Xtr[LEAK_COL].astype("category")
            Xte[LEAK_COL] = Xte[LEAK_COL].astype("category")
        elif severity == "affine":
            Xtr[LEAK_COL], Xte[LEAK_COL] = 2.0 * ctr + 3.0, 2.0 * cte + 3.0
        elif severity == "label_encoded":
            mapping = rng.permutation(codes.max() + 1) * 7 + 2
            Xtr[LEAK_COL] = pd.Categorical([f"code{v}" for v in mapping[ctr]])
            Xte[LEAK_COL] = pd.Categorical([f"code{v}" for v in mapping[cte]])
        else:
            raise ValueError(severity)
        return Xtr, ytr, Xte, {"kind": "column", "column": LEAK_COL}

    if leak_type == "T2":
        if severity.startswith("sigma"):
            s = float(severity[5:]) * float(np.std(codes) or 1.0)
            Xtr[LEAK_COL] = ctr + rng.normal(0, s, ctr.size)
            Xte[LEAK_COL] = cte + rng.normal(0, s, cte.size)
        elif severity.startswith("flip"):
            p = float(severity[4:])
            k = codes.max() + 1

            def flip(c):
                out = c.copy()
                m = rng.rand(c.size) < p
                out[m] = (c[m] + rng.randint(1, k, m.sum())) % k
                return pd.Categorical([f"v{v}" for v in out])

            Xtr[LEAK_COL], Xte[LEAK_COL] = flip(ctr), flip(cte)
        else:
            raise ValueError(severity)
        return Xtr, ytr, Xte, {"kind": "column", "column": LEAK_COL}

    if leak_type == "T3":
        order = np.argsort(ctr, kind="mergesort")
        if severity == "fully_sorted":
            perm = order
        elif severity == "sorted80_blocks":
            perm = order.copy()
            n_move = int(round(0.2 * perm.size))
            movers = rng.choice(perm.size, n_move, replace=False)
            moved = perm[movers]
            rest = np.delete(perm, movers)
            for v in moved:
                rest = np.insert(rest, rng.randint(0, rest.size + 1), v)
            perm = rest
        elif severity == "shuffled_blocks50":
            blocks = [order[i : i + 50] for i in range(0, order.size, 50)]
            perm = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        else:
            raise ValueError(severity)
        Xtr = Xtr.iloc[perm].reset_index(drop=True)
        ytr = ytr.iloc[perm].reset_index(drop=True)
        return Xtr, ytr, Xte, {"kind": "row_order"}

    if leak_type == "T4":
        frac = float(severity.rstrip("%")) / 100.0
        m = max(1, int(np.ceil(frac * len(Xte))))
        pick = rng.choice(len(Xte), m, replace=False)
        Xtr = pd.concat([Xtr, Xte.iloc[pick]], ignore_index=True)
        ytr = pd.concat([ytr, y_test.iloc[pick]], ignore_index=True)
        shuffle = rng.permutation(len(Xtr))
        Xtr = Xtr.iloc[shuffle].reset_index(drop=True)
        ytr = ytr.iloc[shuffle].reset_index(drop=True)
        return Xtr, ytr, Xte, {"kind": "overlap"}

    raise ValueError(leak_type)


def is_detected(report, expected: dict[str, Any]) -> bool:
    kind = expected["kind"]
    if kind == "column":
        return expected["column"] in report.flagged_columns()
    if kind == "row_order":
        return "row_order" in report.flagged_types()
    if kind == "overlap":
        return "train_test_overlap" in report.flagged_types()
    raise ValueError(kind)


def inject_null_column(X: pd.DataFrame, kind: str, seed: int) -> tuple[pd.DataFrame, str]:
    rng = np.random.RandomState(seed)
    n = len(X)
    col = f"__null_{kind}__"
    X2 = X.copy()
    if kind == "normal":
        X2[col] = rng.normal(size=n)
    elif kind == "uniform":
        X2[col] = rng.uniform(size=n)
    elif kind == "categorical":
        X2[col] = pd.Categorical(rng.choice(list("abcdefgh"), n))
    elif kind == "id_like":
        X2[col] = rng.permutation(n).astype(float)
    elif kind == "boolean":
        X2[col] = rng.rand(n) < 0.5
    else:
        raise ValueError(kind)
    return X2, col
