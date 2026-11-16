"""Intervals and verdicts (ARCHITECTURE 4.4)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261116
LEVEL = 0.95


def bootstrap_mean_ci(
    values: NDArray[np.floating],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = LEVEL,
) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean. Returns (mean, low, high)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("no values")
    mean = float(v.mean())
    if v.size == 1:
        return mean, mean, mean
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, v.size, size=(n_resamples, v.size))
    means = v[idx].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return mean, float(min(lo, mean)), float(max(hi, mean))


def verdict(diffs: NDArray[np.floating], **kw: float) -> str:
    """'win' if the CI of the mean difference is above 0, 'loss' if below, else 'tie'.

    `diffs` are oriented so that positive means the first system is better.
    """
    _, lo, hi = bootstrap_mean_ci(diffs, **kw)  # type: ignore[arg-type]
    if lo > 0:
        return "win"
    if hi < 0:
        return "loss"
    return "tie"


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
