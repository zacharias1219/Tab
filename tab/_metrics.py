"""Tab's internal model-selection metrics. Not used for reported scores (evalkit owns those, HF-3)."""

from __future__ import annotations

import numpy as np

CLIP = 1e-15


def log_loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1 - CLIP)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(len(y)), np.asarray(y, dtype=np.int64)])


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss_rows(y, p).mean())


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    idx = np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(total)
