"""In-fold preprocessing (US-2). Every statistic is computed in `fit` from the rows it is given."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

OTHER = "__other__"
MISSING = "__missing__"


def dedupe_columns(columns) -> list[str]:
    """Stringify column names and make duplicates unique deterministically (a, a__dup1, ...)."""
    seen: dict[str, int] = {}
    out = []
    for c in columns:
        name = str(c)
        if name in seen:
            seen[name] += 1
            new = f"{name}__dup{seen[name]}"
            while new in seen:
                seen[name] += 1
                new = f"{name}__dup{seen[name]}"
            seen[new] = 0
            out.append(new)
        else:
            seen[name] = 0
            out.append(name)
    return out


def normalize_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Positional copy with unique string column names. No statistics involved."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    out = X.copy()
    out.columns = dedupe_columns(X.columns)
    return out


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
        return "numeric"
    return "categorical"


def _numeric_values(s: pd.Series, kind: str) -> np.ndarray:
    if kind == "datetime":
        v = s.astype("int64").to_numpy().astype(np.float64)
        v[s.isna().to_numpy()] = np.nan
        return v
    return pd.to_numeric(s.astype("Float64") if pd.api.types.is_bool_dtype(s) else s, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def _cat_values(s: pd.Series) -> np.ndarray:
    isna = s.isna().to_numpy()
    v = s.astype(object).to_numpy()
    out = np.array([MISSING if m else str(x) for x, m in zip(v, isna, strict=True)], dtype=object)
    return out


class Preprocessor:
    """mode='onehot' (linear, MLP): impute median + missing indicator, winsorize, standardize,
    one-hot top levels. mode='ordinal' (trees): impute + indicator, frequency-rank codes."""

    def __init__(self, mode: str = "onehot", max_levels: int | None = None, observer=None):
        if mode not in {"onehot", "ordinal"}:
            raise ValueError(mode)
        self.mode = mode
        self.max_levels = max_levels if max_levels is not None else (20 if mode == "onehot" else 254)
        self.observer = observer

    # -- fit ---------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> Preprocessor:
        X = normalize_frame(X)
        if self.observer is not None:
            self.observer.record("preprocess", X.index)
        num: dict[str, dict[str, Any]] = {}
        cat: dict[str, dict[str, Any]] = {}
        for c in X.columns:
            s = X[c]
            kind = column_kind(s)
            if kind in {"numeric", "datetime"}:
                v = _numeric_values(s, kind)
                finite = v[np.isfinite(v)]
                if finite.size == 0:
                    continue  # all missing: dropped
                lo, hi = np.quantile(finite, [0.001, 0.999])
                clipped = np.clip(finite, lo, hi)
                has_missing = bool(np.isnan(v).any())
                if np.ptp(clipped) == 0 and not has_missing:
                    continue  # constant: dropped
                mean = float(clipped.mean())
                std = float(clipped.std())
                num[c] = {
                    "kind": kind,
                    "median": float(np.median(finite)),
                    "lo": float(lo),
                    "hi": float(hi),
                    "mean": mean,
                    "std": std if std > 0 else 1.0,
                    "indicator": has_missing,
                }
            else:
                v = _cat_values(s)
                vals, counts = np.unique(v, return_counts=True)
                order = sorted(zip(-counts, vals, strict=True))
                levels = [str(val) for _, val in order if val != MISSING][: self.max_levels]
                if len(vals) <= 1:
                    continue  # constant
                cat[c] = {"levels": levels, "index": {lv: i for i, lv in enumerate(levels)}}
        self.columns_ = list(X.columns)
        self.stats_ = {"numeric": num, "categorical": cat}
        self.n_features_out_ = self._width()
        return self

    def _width(self) -> int:
        w = 0
        for st in self.stats_["numeric"].values():
            w += 1 + int(st["indicator"])
        for st in self.stats_["categorical"].values():
            w += (len(st["levels"]) + 2) if self.mode == "onehot" else 1
        return max(w, 1)

    # -- transform ---------------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = normalize_frame(X)
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"columns missing at predict time: {missing[:5]}")
        n = len(X)
        blocks: list[np.ndarray] = []
        for c, st in self.stats_["numeric"].items():
            v = _numeric_values(X[c], st["kind"])
            miss = np.isnan(v)
            v = np.where(np.isnan(v), st["median"], v)
            v = np.clip(v, st["lo"], st["hi"])
            if self.mode == "onehot":
                v = (v - st["mean"]) / st["std"]
            blocks.append(v[:, None])
            if st["indicator"]:
                blocks.append(miss.astype(np.float64)[:, None])
        for c, st in self.stats_["categorical"].items():
            v = _cat_values(X[c])
            idx = st["index"]
            k = len(st["levels"])
            codes = np.array([idx.get(x, k if x != MISSING else k + 1) for x in v], dtype=np.int64)
            if self.mode == "onehot":
                oh = np.zeros((n, k + 2), dtype=np.float64)
                oh[np.arange(n), codes] = 1.0
                blocks.append(oh)
            else:
                blocks.append(codes.astype(np.float64)[:, None])
        if not blocks:
            return np.zeros((n, 1), dtype=np.float64)
        out = np.hstack(blocks)
        return np.ascontiguousarray(out, dtype=np.float64)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def _stats_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_stats_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_stats_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float):
        return isinstance(b, float) and (a == b or (np.isnan(a) and np.isnan(b)))
    return a == b


def fold_clean(
    factory: Callable[..., Preprocessor],
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    pass_peek: bool = False,
    seed: int = 0,
) -> bool:
    """True if fitted statistics do not change when the non-training rows change.

    The check perturbs `X_other` (shuffled values, shifted numerics, new category levels). A
    preprocessor that peeks at those rows produces different statistics and fails.
    """
    rng = np.random.default_rng(seed)
    pert = X_other.copy()
    for c in pert.columns:
        s = pert[c]
        if column_kind(s) == "numeric" and not pd.api.types.is_bool_dtype(s):
            pert[c] = pd.to_numeric(s, errors="coerce") * 3.0 + 1000.0
        else:
            pert[c] = pd.Series([f"perturbed{i % 7}" for i in range(len(s))], index=s.index, dtype=object)
    pert = pert.iloc[rng.permutation(len(pert))]

    if pass_peek:
        a = _with_peek(factory, X_other).fit(X_train)
        b = _with_peek(factory, pert).fit(X_train)
    else:
        a = factory().fit(X_train)
        b = factory().fit(X_train)
    if not _stats_equal(a.stats_, b.stats_):
        return False
    return bool(np.array_equal(a.transform(X_train), b.transform(X_train)))


def _with_peek(factory, peek):
    p = factory()
    p.peek = peek
    return p
