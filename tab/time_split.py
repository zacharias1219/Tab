"""Forward-chaining time splits (US-9). Pure function over a timestamp array."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _to_numeric(ts) -> np.ndarray:
    s = pd.Series(np.asarray(ts) if not isinstance(ts, pd.Series) else ts.to_numpy())
    if pd.api.types.is_datetime64_any_dtype(s):
        if s.isna().any():
            raise ValueError("missing timestamps: time column has NaT values")
        return s.astype("int64").to_numpy().astype(np.float64)
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError("time column must be numeric or datetime")
    arr = s.to_numpy(dtype=np.float64)
    if np.isnan(arr).any():
        raise ValueError("missing timestamps: time column has NaN values")
    return arr


def gap_units(ts, gap) -> float:
    """Gap in the units of the timestamp array (nanoseconds for datetimes if given a Timedelta)."""
    if isinstance(gap, pd.Timedelta):
        return float(gap.value)
    return float(gap)


def forward_chaining_splits(
    ts, n_splits: int = 5, gap: float | pd.Timedelta = 0, y=None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split rows into n_splits (train, validation) pairs of original positions.

    Distinct timestamps are cut into n_splits + 1 contiguous chunks of roughly equal row count; split i
    trains on chunks 0..i and validates on chunk i+1, keeping only validation rows whose timestamp is
    >= max(train timestamp) + gap. Rows sharing a timestamp never straddle a boundary.
    """
    t = _to_numeric(ts)
    g = gap_units(ts, gap)
    if g < 0:
        raise ValueError("gap must be >= 0")
    uniq, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    if uniq.size < n_splits + 1:
        raise ValueError(f"need at least {n_splits + 1} distinct timestamps, got {uniq.size}")
    cum = np.cumsum(counts)
    targets = np.linspace(0, cum[-1], n_splits + 2)[1:-1]
    cuts = np.searchsorted(cum, targets, side="left") + 1  # chunk boundaries in unique-ts index
    cuts = np.clip(cuts, 1, uniq.size - 1)
    cuts = np.unique(cuts)
    if cuts.size < n_splits:
        # fall back to equal counts of distinct timestamps
        cuts = np.linspace(0, uniq.size, n_splits + 2)[1:-1].round().astype(int)
        cuts = np.unique(np.clip(cuts, 1, uniq.size - 1))
        if cuts.size < n_splits:
            raise ValueError("cannot form the requested number of time splits")
    bounds = np.concatenate([[0], cuts, [uniq.size]])
    yv = None if y is None else np.asarray(y)
    splits = []
    for i in range(n_splits):
        train_u = inverse < bounds[i + 1]
        val_u = (inverse >= bounds[i + 1]) & (inverse < bounds[i + 2])
        tr = np.flatnonzero(train_u)
        max_tr = t[tr].max()
        va = np.flatnonzero(val_u & (t >= max_tr + g))
        if va.size == 0:
            raise ValueError(f"gap {gap} is larger than the data allows for split {i}")
        if yv is not None and np.unique(yv[va]).size < 2:
            warnings.warn(f"time split {i}: validation fold has a single class", UserWarning, stacklevel=2)
        splits.append((tr, va))
    return splits
