"""US-10 / HF-1: the meta-learner trains only on out-of-fold predictions."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tab import Tab
from tab.nested import FitObserver
from tab.stacking import check_oof_provenance, oof_predictions
from tests.canary.leaky import leaky_oof


def test_tab_stacker_provenance_is_clean(binary_frame):
    X, y = binary_frame
    obs = FitObserver()
    m = Tab(max_trials=4, seed=0, stacking=True, _observer=obs).fit(X, y)
    assert obs.provenance, "stacker must record OOF provenance"
    check_oof_provenance(obs.provenance)
    assert m.report_.stack_considered


def _fit_fn(X, y):
    def fit_rows(rows):
        model = LogisticRegression().fit(X[rows], y[rows])
        return model.predict_proba

    return fit_rows


def test_oof_predictions_provenance_clean():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    folds = [(np.setdiff1d(np.arange(120), v), v) for v in np.array_split(rng.permutation(120), 3)]
    preds, records = oof_predictions(_fit_fn(X, y), X, folds, k=2)
    assert preds.shape == (120, 2)
    check_oof_provenance(records)


def test_canary_in_sample_predictions_are_caught():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    folds = [(np.setdiff1d(np.arange(120), v), v) for v in np.array_split(rng.permutation(120), 3)]
    _, records = leaky_oof(_fit_fn(X, y), X, y, folds)
    with pytest.raises(AssertionError):
        check_oof_provenance(records)


def test_stacking_needs_two_base_models(binary_frame):
    X, y = binary_frame
    m = Tab(max_trials=1, seed=0, stacking=True).fit(X, y)
    assert m.report_.chosen != "stack"
