"""Scoring definitions (ARCHITECTURE 4.4).

Labels are integer codes into a fixed class list; probability columns follow the same list.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

CLIP = 1e-15
ECE_BINS = 15

FloatArr = NDArray[np.float64]
IntArr = NDArray[np.int64]


def align_proba(
    proba: NDArray[np.floating], proba_classes: Sequence[str], all_classes: Sequence[str]
) -> FloatArr:
    """Reorder columns of `proba` (in `proba_classes` order) to `all_classes` order.

    Classes the system never saw get a zero column (the clip in log_loss then applies).
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != len(proba_classes):
        raise ValueError(f"proba shape {proba.shape} does not match {len(proba_classes)} classes")
    index = {c: i for i, c in enumerate(all_classes)}
    unknown = [c for c in proba_classes if c not in index]
    if unknown:
        raise ValueError(f"predicted classes not in the dataset's class list: {unknown}")
    if len(set(proba_classes)) != len(proba_classes):
        raise ValueError("duplicate class names in proba_classes")
    out = np.zeros((proba.shape[0], len(all_classes)), dtype=np.float64)
    for j, c in enumerate(proba_classes):
        out[:, index[c]] = proba[:, j]
    return out


def check_proba(proba: NDArray[np.floating], tol: float = 1e-6) -> None:
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("proba must be 2-D")
    if not np.all(np.isfinite(p)):
        raise ValueError("proba contains NaN or inf")
    if np.any(p < -tol) or np.any(p > 1 + tol):
        raise ValueError("proba outside [0, 1]")
    if np.max(np.abs(p.sum(axis=1) - 1.0)) > tol:
        raise ValueError("proba rows do not sum to 1")


def log_loss(y: NDArray[np.integer], proba: NDArray[np.floating]) -> float:
    """Mean negative log likelihood, probabilities clipped to [1e-15, 1 - 1e-15]."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    if p.shape[0] != y.shape[0]:
        raise ValueError("length mismatch")
    p = np.clip(p, CLIP, 1 - CLIP)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(y.size), y])))


def prior_proba(y_train: NDArray[np.integer], k: int, n: int) -> FloatArr:
    """Class-prior predictor: training-fold class frequencies, repeated for n rows."""
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=k).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError("empty training labels")
    return np.tile(counts / counts.sum(), (n, 1))


def skill(model_log_loss: float, prior_log_loss: float) -> float:
    """1 - model / prior. 0 = no better than class prior, 1 = perfect."""
    if not prior_log_loss > 0:
        raise ValueError("prior log loss must be positive")
    return float(1.0 - model_log_loss / prior_log_loss)


def roc_auc_binary(y: NDArray[np.integer], score: NDArray[np.floating]) -> float | None:
    """Mann-Whitney AUC with average ranks for ties. None if one class is absent."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece(y: NDArray[np.integer], proba: NDArray[np.floating], n_bins: int = ECE_BINS) -> float:
    """Expected calibration error on the max-class probability, equal-width bins."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    # bin i covers (i/n, (i+1)/n]; conf == 0 cannot happen for a max over >= 2 classes summing to 1
    idx = np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += abs(float(correct[m].mean()) - float(conf[m].mean())) * (m.sum() / y.size)
    return float(total)
