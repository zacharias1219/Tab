"""The boring model zoo (US-4): logistic, random forest, histogram gradient boosting, small MLP."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class Family:
    name: str
    preproc_mode: str
    sample: Callable[[np.random.RandomState, int, int], dict[str, Any]]
    default: Callable[[int, int], dict[str, Any]]
    cost: Callable[[dict[str, Any]], float] = field(default=lambda p: 1.0)


def _logistic_sample(rng, n, d):
    return {"C": float(10 ** rng.uniform(-3, 2))}


def _rf_sample(rng, n, d):
    return {
        "n_estimators": int(rng.choice([100, 200, 300])),
        "max_features": rng.choice(["sqrt", "0.3", "0.6"]).item(),
        "min_samples_leaf": int(rng.choice([1, 2, 5, 10])),
    }


def _hgb_sample(rng, n, d):
    return {
        "learning_rate": float(10 ** rng.uniform(np.log10(0.02), np.log10(0.3))),
        "max_leaf_nodes": int(rng.choice([7, 15, 31, 63, 127])),
        "min_samples_leaf": int(rng.choice([5, 10, 20, 50, 100])),
        "l2_regularization": float(rng.choice([0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0])),
    }


def _mlp_sample(rng, n, d):
    return {
        "hidden": rng.choice(["64", "128", "64,64", "128,64"]).item(),
        "alpha": float(10 ** rng.uniform(-5, -1)),
        "learning_rate_init": float(10 ** rng.uniform(-3.5, -2)),
    }


FAMILIES: dict[str, Family] = {
    "logistic": Family(
        "logistic", "onehot", _logistic_sample, lambda n, d: {"C": 1.0}, lambda p: 1.0
    ),
    "random_forest": Family(
        "random_forest",
        "ordinal",
        _rf_sample,
        lambda n, d: {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1},
        lambda p: p["n_estimators"] / 100.0,
    ),
    "hist_gb": Family(
        "hist_gb",
        "ordinal",
        _hgb_sample,
        lambda n, d: {
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
        lambda p: 0.1 / p["learning_rate"] * (p["max_leaf_nodes"] / 31.0) ** 0.5,
    ),
    "mlp": Family(
        "mlp",
        "onehot",
        _mlp_sample,
        lambda n, d: {"hidden": "64", "alpha": 1e-4, "learning_rate_init": 1e-3},
        lambda p: sum(int(h) for h in p["hidden"].split(",")) / 64.0,
    ),
}

# Fixed order of the first trials: cheap, robust defaults first so a model exists early.
DEFAULT_ORDER = ["logistic", "hist_gb", "random_forest", "mlp"]


def default_config(family: str, n_rows: int, n_features: int) -> dict[str, Any]:
    return {"family": family, "params": FAMILIES[family].default(n_rows, n_features)}


def sample_config(rng: np.random.RandomState, n_rows: int, n_features: int) -> dict[str, Any]:
    family = DEFAULT_ORDER[int(rng.randint(len(DEFAULT_ORDER)))]
    return {"family": family, "params": FAMILIES[family].sample(rng, n_rows, n_features)}


def _max_features(v: str):
    return v if v == "sqrt" else float(v)


def build_model(cfg: dict[str, Any], seed: int, n_jobs: int = 1):
    fam, p = cfg["family"], cfg["params"]
    if fam == "logistic":
        return LogisticRegression(C=p["C"], max_iter=1000, random_state=seed)
    if fam == "random_forest":
        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_features=_max_features(p["max_features"]),
            min_samples_leaf=p["min_samples_leaf"],
            n_jobs=n_jobs,
            random_state=seed,
        )
    if fam == "hist_gb":
        return HistGradientBoostingClassifier(
            learning_rate=p["learning_rate"],
            max_leaf_nodes=p["max_leaf_nodes"],
            min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"],
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
    if fam == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(int(h) for h in p["hidden"].split(",")),
            alpha=p["alpha"],
            learning_rate_init=p["learning_rate_init"],
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed,
        )
    raise ValueError(f"unknown family {fam}")


def fit_quiet(model, X: np.ndarray, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            model.fit(X, y)
        except ValueError:
            # Early stopping makes a stratified validation split; a class with one member breaks it.
            if getattr(model, "early_stopping", False) is True:
                model.set_params(early_stopping=False)
                model.fit(X, y)
            else:
                raise
    return model


def full_proba(model, X: np.ndarray, k: int) -> np.ndarray:
    """predict_proba mapped onto all k classes (a fold may miss a class), rows renormalized."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = model.predict_proba(X)
    out = np.zeros((X.shape[0], k), dtype=np.float64)
    out[:, np.asarray(model.classes_, dtype=np.int64)] = p
    out = np.clip(out, 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)
