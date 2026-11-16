"""Configuration and envelope limits."""

from __future__ import annotations

from dataclasses import dataclass

MAX_ROWS = 50_000
MAX_COLS = 200
MAX_CLASSES = 10

# Provisional until bench/tune_dev.py sets them on the development pool (PREREG section 5).
PROVISIONAL = True


@dataclass(frozen=True)
class ControllerConfig:
    """Plateau stopping: stop when the best inner log loss improved by less than `delta`
    (relative) over the last `window` trials. The ceiling always wins."""

    window: int = 8
    delta: float = 0.001
    enabled: bool = True
    min_trials: int = 1

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if not 0 <= self.delta < 1:
            raise ValueError("delta must be in [0, 1)")
