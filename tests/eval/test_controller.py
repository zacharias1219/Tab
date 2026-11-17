"""US-7: budget controller on the 8 pre-specified synthetic curves. Pure, runs in milliseconds."""

import numpy as np
import pytest

from tab.controller import ControllerConfig, should_stop, simulate

CFG = ControllerConfig(window=8, delta=0.001)


def _check_invariants(curve, res, ceiling_s, trial_s):
    assert res.elapsed_s <= ceiling_s + 1e-9, "ceiling exceeded"
    seen = np.asarray(curve[: res.n_trials], float)
    assert res.best_index == int(np.nanargmin(seen)), "best-so-far not returned"
    assert res.best_loss == pytest.approx(float(np.nanmin(seen)))
    assert res.n_trials * trial_s <= ceiling_s + 1e-9


def test_plateau_early():
    curve = [1.0, 0.8, 0.6, 0.5, 0.45] + [0.46 + 0.001 * i for i in range(95)]
    res = simulate(curve, CFG, trial_s=1.0, ceiling_s=1000)
    assert res.stop_reason == "plateau"
    assert res.n_trials == 5 + CFG.window
    _check_invariants(curve, res, 1000, 1.0)


def test_plateau_late():
    curve = [0.99**i for i in range(40)] + [0.99**39] * 60
    res = simulate(curve, CFG, trial_s=1.0, ceiling_s=1000)
    assert res.stop_reason == "plateau"
    assert 40 < res.n_trials <= 40 + CFG.window
    _check_invariants(curve, res, 1000, 1.0)


def test_noisy_plateau():
    rng = np.random.default_rng(0)
    curve = [1.0, 0.7, 0.55] + list(0.5 + np.abs(rng.normal(0, 0.05, 200)))
    res = simulate(curve, CFG, trial_s=1.0, ceiling_s=1000)
    assert res.stop_reason == "plateau"
    assert res.n_trials < 150
    _check_invariants(curve, res, 1000, 1.0)


def test_steady_descent_runs_to_ceiling():
    curve = [0.98**i for i in range(500)]
    res = simulate(curve, CFG, trial_s=1.0, ceiling_s=100)
    assert res.stop_reason == "ceiling"
    assert res.n_trials == 100
    _check_invariants(curve, res, 100, 1.0)


def test_rising_overfit_returns_early_best():
    curve = [1.0, 0.8, 0.7] + [0.7 + 0.05 * i for i in range(1, 50)]
    res = simulate(curve, CFG, trial_s=1.0, ceiling_s=1000)
    assert res.stop_reason == "plateau"
    assert res.best_index == 2
    assert res.n_trials == 3 + CFG.window
    _check_invariants(curve, res, 1000, 1.0)


def test_flat_from_trial_one():
    curve = [0.5] * 50
    res = simulate(curve, CFG, trial_s=1.0, ceiling_s=1000)
    assert res.stop_reason == "plateau"
    assert res.n_trials == CFG.window + 1
    assert res.best_index == 0
    _check_invariants(curve, res, 1000, 1.0)


def test_sawtooth_never_plateaus():
    curve = []
    low = 1.0
    for i in range(200):
        if i % 4 == 0:
            low *= 0.95
            curve.append(low)
        else:
            curve.append(low * 1.3)
    res = simulate(curve, CFG, trial_s=1.0, ceiling_s=120)
    assert res.stop_reason == "ceiling"
    _check_invariants(curve, res, 120, 1.0)


def test_single_trial():
    curve = [0.5]
    res = simulate(curve, CFG, trial_s=1.0, ceiling_s=1.5)
    assert res.n_trials == 1
    assert res.best_index == 0
    assert res.stop_reason == "ceiling"
    _check_invariants(curve, res, 1.5, 1.0)


def test_ceiling_wins_when_controller_disabled():
    curve = [0.5] * 50
    res = simulate(curve, ControllerConfig(enabled=False), trial_s=1.0, ceiling_s=20)
    assert res.stop_reason == "ceiling"
    assert res.n_trials == 20


def test_failed_trial_counts_as_no_improvement():
    curve = [0.5, float("nan")] + [0.5] * 20
    res = simulate(curve, CFG, trial_s=1.0, ceiling_s=1000)
    assert res.best_index == 0
    assert res.stop_reason == "plateau"


def test_should_stop_is_pure():
    hist = [1.0] * 20
    assert should_stop(hist, CFG) == should_stop(list(hist), CFG) == "plateau"
    assert should_stop([1.0] * CFG.window, CFG) is None
