"""Budget controller (US-7). Pure functions over loss histories, no I/O, no clock (ADR-007)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tab.config import ControllerConfig


def best_so_far(losses: Sequence[float]) -> list[float]:
    out: list[float] = []
    best = math.inf
    for v in losses:
        if v is not None and not math.isnan(v) and v < best:
            best = v
        out.append(best)
    return out


def should_stop(losses: Sequence[float], cfg: ControllerConfig) -> str | None:
    """Return 'plateau' if the relative improvement of the best loss over the last
    `cfg.window` trials is below `cfg.delta`, else None. `losses` is the raw per-trial history."""
    if not cfg.enabled or len(losses) <= cfg.window or len(losses) < cfg.min_trials:
        return None
    best = best_so_far(losses)
    before, now = best[-cfg.window - 1], best[-1]
    if math.isinf(now):
        return None
    if math.isinf(before):
        return None
    rel = (before - now) / max(abs(before), 1e-12)
    return "plateau" if rel < cfg.delta else None


@dataclass(frozen=True)
class SimResult:
    n_trials: int
    best_index: int
    best_loss: float
    stop_reason: str
    elapsed_s: float


def simulate(
    curve: Sequence[float], cfg: ControllerConfig, trial_s: float, ceiling_s: float
) -> SimResult:
    """Replay a loss curve under a ceiling: a trial starts only if it can finish inside the ceiling."""
    history: list[float] = []
    elapsed = 0.0
    reason = "exhausted"
    for v in curve:
        if elapsed + trial_s > ceiling_s + 1e-12:
            reason = "ceiling"
            break
        elapsed += trial_s
        history.append(float(v))
        r = should_stop(history, cfg)
        if r:
            reason = r
            break
    else:
        if elapsed + trial_s > ceiling_s + 1e-12:
            reason = "ceiling"
    if not history:
        raise ValueError("ceiling too small for a single trial")
    clean = [math.inf if math.isnan(v) else v for v in history]
    idx = min(range(len(clean)), key=lambda i: (clean[i], i))
    return SimResult(len(history), idx, clean[idx], reason, elapsed)
