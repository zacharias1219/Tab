"""Leakage detector (US-8): scored flags for target copy, near-copy, row-order / ID leak and
train/test overlap, plus a false-positive rate measured on injected null columns.

Every statistic is a cheap, deterministic function of the table. It can and will flag legitimately
strong features; flags are for review, not automatic removal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from tab.preprocess import _cat_values, column_kind, normalize_frame

DEFAULT_THRESHOLD = 0.9  # single-column skill; provisional until set on the development pool
RUNS_Z_THRESHOLD = 6.0
COPY_SKILL = 0.98
N_BINS = 20
N_FOLDS = 5
SMOOTH = 0.5


@dataclass
class LeakReport:
    flags: list[dict[str, Any]] = field(default_factory=list)
    row_order: dict[str, Any] = field(default_factory=dict)
    overlap: dict[str, Any] | None = None
    null_fpr: float | None = None
    null_n: int = 0
    threshold: float = DEFAULT_THRESHOLD

    def flagged_columns(self) -> list[str]:
        return [f["column"] for f in self.flags if f["flagged"] and f["column"] is not None]

    def flagged_types(self) -> set[str]:
        return {f["type"] for f in self.flags if f["flagged"]}

    def to_dict(self) -> dict[str, Any]:
        return {
            "flags": self.flags,
            "row_order": self.row_order,
            "overlap": self.overlap,
            "null_fpr": self.null_fpr,
            "null_n": self.null_n,
            "threshold": self.threshold,
        }


def _codes(y: pd.Series) -> tuple[np.ndarray, int]:
    vals = pd.Series(y).astype(str).to_numpy()
    uniq, inv = np.unique(vals, return_inverse=True)
    return inv.astype(np.int64), uniq.size


def _column_values(s: pd.Series) -> tuple[np.ndarray, bool]:
    """Return (values, discrete). Discrete columns are integer group ids; continuous are floats
    with NaN mapped to a sentinel below the minimum (missingness can itself be a leak)."""
    kind = column_kind(s)
    if kind == "categorical":
        v = _cat_values(s)
        uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        if uniq.size > 50:
            top = np.argsort(-counts, kind="mergesort")[:50]
            remap = np.full(uniq.size, 50)
            remap[top] = np.arange(top.size)
            inv = remap[inv]
        return inv.astype(np.int64), True
    if kind == "datetime":
        arr = s.astype("int64").astype(float).where(~s.isna()).to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        v = pd.to_numeric(s.astype("Float64") if pd.api.types.is_bool_dtype(s) else s, errors="coerce")
        arr = v.to_numpy(dtype=np.float64, na_value=np.nan)
    nan = np.isnan(arr)
    finite = arr[~nan]
    if finite.size == 0:
        return np.zeros(arr.size, dtype=np.int64), True
    uniq = np.unique(finite)
    if uniq.size <= 50:
        out = np.zeros(arr.size, dtype=np.int64)
        out[~nan] = np.searchsorted(uniq, finite) + 1
        return out, True
    lo = float(finite.min())
    span = float(finite.max() - lo) or 1.0
    arr = np.where(nan, lo - span, arr)
    return arr, False


def _fold_groups(values: np.ndarray, discrete: bool, y: np.ndarray, tr: np.ndarray, te: np.ndarray, n_groups_cap: int = N_BINS):
    if discrete:
        return values[tr], values[te], int(values.max()) + 1
    tree = DecisionTreeClassifier(
        max_leaf_nodes=n_groups_cap, min_samples_leaf=max(3, int(tr.sum()) // 200), random_state=0
    )
    tree.fit(values[tr, None], y[tr])
    gtr = tree.apply(values[tr, None])
    gte = tree.apply(values[te, None])
    return gtr, gte, int(tree.tree_.node_count)


def _groups(s: pd.Series) -> np.ndarray:
    """Discretize a column on all rows (used for purity): groups, or 20 quantile bins."""
    values, discrete = _column_values(s)
    if discrete:
        return values
    edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def _fold_ids(n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.permutation(n) % N_FOLDS


def single_column_skill(values: np.ndarray, discrete: bool, y: np.ndarray, k: int, folds: np.ndarray) -> float:
    """Cross-validated skill of a one-column predictor: per-group class frequencies smoothed toward the
    prior (groups = categories, or leaves of a depth-limited 1-D tree for continuous columns)."""
    ll = 0.0
    prior_ll = 0.0
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        if not te.any() or not tr.any():
            continue
        gtr, gte, n_groups = _fold_groups(values, discrete, y, tr, te)
        counts = np.zeros((n_groups, k))
        np.add.at(counts, (gtr, y[tr]), 1.0)
        prior = np.bincount(y[tr], minlength=k) + 1.0
        prior = prior / prior.sum()
        probs = (counts + SMOOTH * prior[None, :]) / (counts.sum(1, keepdims=True) + SMOOTH)
        p = probs[gte, y[te]]
        ll += -np.log(np.clip(p, 1e-15, 1)).sum()
        prior_ll += -np.log(np.clip(prior[y[te]], 1e-15, 1)).sum()
    if prior_ll <= 0:
        return 0.0
    return float(1.0 - ll / prior_ll)


def _purity(g: np.ndarray, y: np.ndarray, k: int) -> float:
    """Fraction of rows whose group is single-class (1.0 means the column determines the label)."""
    n_groups = int(g.max()) + 1
    counts = np.zeros((n_groups, k))
    np.add.at(counts, (g, y), 1.0)
    pure = (counts > 0).sum(1) == 1
    return float(counts[pure].sum() / len(y))


def _runs_z(y: np.ndarray, k: int) -> float:
    """z-score of the number of adjacent equal labels vs a random row order."""
    n = y.size
    if n < 3:
        return 0.0
    counts = np.bincount(y, minlength=k).astype(float)
    p = float((counts * (counts - 1)).sum() / (n * (n - 1)))
    obs = float((y[1:] == y[:-1]).sum())
    exp = (n - 1) * p
    sd = np.sqrt(max((n - 1) * p * (1 - p), 1e-12))
    return (obs - exp) / sd


def _row_hashes(X: pd.DataFrame) -> np.ndarray:
    Xn = X.astype(object).where(pd.notna(X), "__nan__")
    return np.array(
        [hashlib.sha1("\x1f".join(map(str, r)).encode()).hexdigest() for r in Xn.itertuples(index=False, name=None)]  # noqa: S324
    )


def _overlap(X: pd.DataFrame, X_test: pd.DataFrame, seed: int) -> dict[str, Any]:
    cols = [c for c in X.columns if c in X_test.columns]
    htr = _row_hashes(X[cols])
    hte = _row_hashes(X_test[cols])
    observed = float(np.isin(hte, htr).mean()) if hte.size else 0.0
    # natural duplicate rate: second half of a random split of train found in the first half
    rng = np.random.RandomState(seed)
    perm = rng.permutation(htr.size)
    a, b = htr[perm[: htr.size // 2]], htr[perm[htr.size // 2 :]]
    base = float(np.isin(b, a).mean()) * 2.0 if b.size else 0.0
    base = min(base, 1.0)
    n = max(hte.size, 1)
    thresh = base + 3.0 * np.sqrt(max(base * (1 - base), 0.0) / n) + 0.005
    return {
        "observed_rate": observed,
        "expected_rate": base,
        "threshold": float(thresh),
        "n_test": int(hte.size),
        "flagged": bool(observed > thresh),
    }


def _null_columns(n: int, m: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed + 1)
    makers = [
        lambda: rng.normal(size=n),
        lambda: rng.uniform(size=n),
        lambda: pd.Categorical(rng.choice(list("abcdefgh"), n)),
        lambda: rng.permutation(n).astype(float),
        lambda: rng.rand(n) < 0.5,
    ]
    return pd.DataFrame({f"__null{i}__": makers[i % len(makers)]() for i in range(m)})


def _scan_column(s: pd.Series, y: np.ndarray, k: int, folds: np.ndarray, threshold: float) -> dict[str, Any]:
    values, discrete = _column_values(s)
    sk = single_column_skill(values, discrete, y, k, folds)
    purity = _purity(_groups(s), y, k)
    n_unique = int(pd.Series(s).nunique(dropna=True))
    id_like = n_unique >= 0.95 * len(s) and column_kind(s) != "categorical"
    if sk >= COPY_SKILL or (purity >= 0.999 and n_unique <= max(k, 2) * 4 and sk >= threshold):
        typ = "target_copy"
    elif id_like:
        typ = "id_order"
    else:
        typ = "near_copy"
    return {
        "column": str(s.name),
        "type": typ,
        "score": sk,
        "purity": purity,
        "flagged": bool(sk >= threshold),
    }


def detect_leaks(
    X: pd.DataFrame,
    y,
    X_test: pd.DataFrame | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    null_columns: int = 5,
    seed: int = 0,
    max_rows: int | None = None,
) -> LeakReport:
    X = normalize_frame(X)
    y = pd.Series(np.asarray(y))
    if max_rows is not None and len(X) > max_rows:
        keep = np.sort(np.random.RandomState(seed).choice(len(X), max_rows, replace=False))  # order kept
        X = X.iloc[keep]
        y = y.iloc[keep]
    yc, k = _codes(y)
    n = len(X)
    folds = _fold_ids(n, seed)
    rep = LeakReport(threshold=threshold)
    for c in X.columns:
        rep.flags.append(_scan_column(X[c], yc, k, folds, threshold))

    # T3: row order. Position as a feature, and adjacent-label runs.
    pos = pd.Series(np.arange(n, dtype=float), name="__row_position__")
    pos_skill = single_column_skill(*_column_values(pos), yc, k, folds)
    z = _runs_z(yc, k)
    flagged = bool(pos_skill >= threshold * 0.5 or z >= RUNS_Z_THRESHOLD)
    rep.row_order = {"position_skill": pos_skill, "runs_z": z, "flagged": flagged}
    rep.flags.append(
        {"column": None, "type": "row_order", "score": float(max(pos_skill, min(z / (2 * RUNS_Z_THRESHOLD), 1.0))),
         "flagged": flagged}
    )

    # T4: overlap with the test table
    if X_test is not None:
        ov = _overlap(X, normalize_frame(X_test), seed)
        rep.overlap = ov
        rep.flags.append({"column": None, "type": "train_test_overlap", "score": ov["observed_rate"], "flagged": ov["flagged"]})

    # false-positive rate on null columns
    if null_columns > 0:
        nulls = _null_columns(n, null_columns, seed)
        fp = sum(_scan_column(nulls[c], yc, k, folds, threshold)["flagged"] for c in nulls.columns)
        rep.null_fpr = fp / null_columns
        rep.null_n = null_columns
    return rep
