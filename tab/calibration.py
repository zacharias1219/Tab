"""Temperature-scaling calibration fitted on inner out-of-fold predictions (US-5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from tab._metrics import ece, log_loss


def _apply(p: np.ndarray, t: float) -> np.ndarray:
    logp = np.log(np.clip(p, 1e-15, 1.0)) / t
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaler:
    """One scalar T: p_cal ∝ p ** (1/T). T > 1 softens, T < 1 sharpens."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        res = minimize_scalar(
            lambda lt: log_loss(y, _apply(p, float(np.exp(lt)))),
            bounds=(np.log(0.05), np.log(20.0)),
            method="bounded",
            options={"xatol": 1e-4},
        )
        self.temperature_ = float(np.exp(res.x))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return _apply(p, self.temperature_)


@dataclass(frozen=True)
class CalibrationResult:
    use: bool
    temperature: float
    ece_before: float
    ece_after: float
    log_loss_before: float
    log_loss_after: float


def cross_fit_calibration(
    p: np.ndarray, y: np.ndarray, folds: np.ndarray, observer=None, index=None
) -> CalibrationResult:
    """Honest before/after numbers: for each inner fold, fit T on the other folds' OOF rows and
    apply it to this fold. `use` is True only if cross-fitted log loss improves.
    The returned temperature is fitted on all rows given (all of them inner-training rows)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    cal = np.empty_like(p)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0:
            cal[te] = p[te]
            continue
        cal[te] = TemperatureScaler().fit(p[tr], y[tr]).transform(p[te])
    full = TemperatureScaler().fit(p, y)
    if observer is not None and index is not None:
        observer.record("calibrator", index)
    ll_b, ll_a = log_loss(y, p), log_loss(y, cal)
    return CalibrationResult(
        use=bool(ll_a < ll_b),
        temperature=full.temperature_,
        ece_before=ece(y, p),
        ece_after=ece(y, cal),
        log_loss_before=ll_b,
        log_loss_after=ll_a,
    )


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return _apply(p, t)
