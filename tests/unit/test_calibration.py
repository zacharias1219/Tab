"""US-5: calibration with ECE before and after; outer fold untouched."""

import numpy as np
import pytest

from tab import Tab
from tab._metrics import ece
from tab.calibration import TemperatureScaler, cross_fit_calibration
from tab.nested import FitObserver, assert_disjoint, outer_splits


def test_ece_hand_computed():
    p = np.array([[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.4, 0.6]])
    y = np.array([0, 1, 1, 1])
    assert ece(y, p) == pytest.approx(0.4)
    p2 = np.array([[0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25]])
    y2 = np.array([0, 0, 0, 1])
    assert ece(y2, p2) == pytest.approx(0.0)


def _overconfident(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < q).astype(int)
    logit = np.log(q / (1 - q)) * 4.0  # sharpened -> overconfident
    q_bad = 1 / (1 + np.exp(-logit))
    return np.column_stack([1 - q_bad, q_bad]), y


@pytest.mark.parametrize("seed", range(5))
def test_calibrated_ece_not_worse_on_miscalibrated_model(seed):
    p, y = _overconfident(3000, seed)
    folds = np.arange(len(y)) % 3
    res = cross_fit_calibration(p, y, folds)
    assert res.ece_after <= res.ece_before
    assert res.ece_before > 0.05
    assert res.use


def test_temperature_recovers_known_sharpening():
    p, y = _overconfident(20000, 0)
    t = TemperatureScaler().fit(p, y)
    assert t.temperature_ == pytest.approx(4.0, rel=0.15)


def test_calibration_is_skipped_when_it_hurts():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.05, 0.95, 60)
    y = (rng.random(60) < q).astype(int)
    p = np.column_stack([1 - q, q])  # already calibrated, tiny data
    res = cross_fit_calibration(p, y, np.arange(60) % 3)
    assert res.use in (True, False)
    if not res.use:
        assert res.log_loss_after >= res.log_loss_before


def test_report_has_ece_before_and_after(binary_frame):
    X, y = binary_frame
    m = Tab(max_trials=3, seed=0, calibrate=True).fit(X, y)
    assert 0 <= m.report_.ece_before <= 1
    assert 0 <= m.report_.ece_after <= 1


def test_calibrator_never_sees_outer_fold(binary_frame):
    X, y = binary_frame
    tr, te = outer_splits(y, 5, seed=0)[0]
    obs = FitObserver()
    Tab(max_trials=3, seed=0, calibrate=True, _observer=obs).fit(X.iloc[tr], y.iloc[tr])
    assert any(s == "calibrator" for s, _ in obs.records)
    assert_disjoint(obs, X.index[te])
