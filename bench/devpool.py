"""Development pool draw (PREREG section 5)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

DEV_SEED = 20261118


def draw_dev_pool(
    eligible: Iterable[int], drawn: Iterable[int], n: int = 12, seed: int = DEV_SEED
) -> list[int]:
    """Undrawn eligible task ids, sorted ascending, then `RandomState(seed).choice(ids, n, replace=False)`.
    Returned sorted ascending."""
    undrawn = sorted(set(int(t) for t in eligible) - set(int(t) for t in drawn))
    if len(undrawn) < n:
        raise ValueError(f"only {len(undrawn)} undrawn tasks, need {n}")
    rng = np.random.RandomState(seed)
    pick = rng.choice(np.array(undrawn), size=n, replace=False)
    return sorted(int(t) for t in pick)
