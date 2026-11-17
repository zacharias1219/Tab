"""Shared fixtures. Builders here produce minimal *valid* documents; tests mutate them."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

SHA = "a" * 64


def _scores() -> dict:
    return {
        "scored_by": "evalkit@0.1.0",
        "log_loss": 0.5,
        "prior_log_loss": 0.69,
        "skill": 1 - 0.5 / 0.69,
        "auc": 0.8,
        "ece": 0.05,
        "n_test": 100,
    }


def make_run(**over) -> dict:
    run = {
        "run_id": "tab__credit-g__f0__s0",
        "system": "tab",
        "dataset": "credit-g",
        "task_id": 31,
        "fold": 0,
        "seed": 0,
        "budget_s": 60.0,
        "actual_wall_s": 58.2,
        "peak_rss_gb_tree": 0.4,
        "n_cpus": 4,
        "status": "ok",
        "error": None,
        "exit_code": 0,
        "proba_sha256": SHA,
        "budget_ratio": 0.97,
        "within_nfr1": True,
        "stop_reason": "ceiling",
        "n_trials": 12,
        "scores": _scores(),
    }
    run.update(over)
    return run


def make_results(**over) -> dict:
    doc = {
        "schema_version": "1",
        "project": "3A-tab",
        "git_sha": "abcdef1",
        "created_utc": "2026-11-16T12:00:00Z",
        "hardware": {
            "cpu_model": "test",
            "n_logical_cpus": 4,
            "ram_gb": 16.0,
            "os": "Linux",
            "python": "3.12.0",
        },
        "versions": {
            "tab": "0.1.0",
            "evalkit": "0.1.0",
            "numpy": "2",
            "pandas": "2",
            "scikit-learn": "1",
        },
        "dataset_list_sha256": SHA,
        "config": {
            "mode": "skeleton",
            "budget_s": 60.0,
            "folds": [0],
            "systems": ["tab"],
            "datasets": ["credit-g"],
        },
        "runs": [make_run()],
        "analyses": {
            "per_dataset": [],
            "nfr1": {
                "system": "tab",
                "n_runs": 1,
                "n_within": 1,
                "fraction_within": 1.0,
                "passes": True,
                "overruns": [],
            },
            "cost": [],
        },
        "headline": {
            "metric": "skill_score_log_loss",
            "system": "tab",
            "is_claim": False,
            "statement": "n = 1, not yet a claim.",
            "comparisons": [],
        },
    }
    doc.update(over)
    return copy.deepcopy(doc)


@pytest.fixture
def valid_results() -> dict:
    return make_results()


def mixed_frame(n: int = 300, seed: int = 0, k: int = 2) -> tuple[pd.DataFrame, pd.Series]:
    """Small mixed-dtype table with signal, missing values, bool and categorical columns."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-2, 2, size=n)
    cat = rng.choice(["a", "b", "c", "d"], size=n)
    flag = rng.random(n) < 0.5
    logits = 1.5 * x1 - x2 + (cat == "a") * 1.0 + flag * 0.5
    if k == 2:
        y = (logits + rng.normal(scale=0.7, size=n) > 0).astype(int)
        labels = np.array(["neg", "pos"])[y]
    else:
        cuts = np.quantile(logits, np.linspace(0, 1, k + 1)[1:-1])
        y = np.digitize(logits + rng.normal(scale=0.5, size=n), cuts)
        labels = np.array([f"c{i}" for i in range(k)])[y]
    x1 = x1.copy()
    x1[rng.random(n) < 0.1] = np.nan
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "cat": pd.Categorical(np.where(rng.random(n) < 0.05, None, cat)),
            "flag": flag,
            "noise": rng.normal(size=n),
        }
    )
    return X, pd.Series(labels, name="y")


@pytest.fixture
def binary_frame():
    return mixed_frame(300, 0, 2)


@pytest.fixture
def multi_frame():
    return mixed_frame(400, 1, 3)
