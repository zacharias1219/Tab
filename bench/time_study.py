"""Time-aware study (PREREG section 8): random folds vs forward chaining, on electricity if its
time ordering holds, and on a synthetic drift set with a known onset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bench import datasets
from evalkit.score import score_predictions
from evalkit.stats import bootstrap_mean_ci
from tab import Tab
from tab.nested import outer_splits
from tab.time_split import forward_chaining_splits

N_SPLITS = 5


def check_ordering(X: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in X.columns:
        return {"usable": False, "reason": f"no column {col!r}"}
    t = pd.to_numeric(X[col], errors="coerce")
    if t.isna().any():
        return {"usable": False, "reason": "missing timestamps"}
    frac_monotone = float((np.diff(t.to_numpy()) >= 0).mean())
    n_unique = int(t.nunique())
    usable = frac_monotone >= 0.99 and n_unique >= 50
    return {
        "usable": usable,
        "fraction_non_decreasing": frac_monotone,
        "n_unique": n_unique,
        "reason": "time column is non-decreasing in row order"
        if usable
        else "row order is not time order",
    }


def _skill(model_fn, X, y, splits, classes) -> list[float]:
    out = []
    for tr, te in splits:
        m = model_fn().fit(X.iloc[tr], y.iloc[tr])
        s = score_predictions(
            m.predict_proba(X.iloc[te]),
            [str(c) for c in m.classes_],
            y.iloc[te].astype(str).to_numpy(),
            y.iloc[tr].astype(str).to_numpy(),
            classes,
        )
        out.append(s["skill"])
    return out


def compare(
    X: pd.DataFrame, y: pd.Series, time_col: str, budget: float, cpus: int, seed: int = 0
) -> dict[str, Any]:
    y = y.astype(str)
    classes = sorted(y.unique())
    order = np.argsort(pd.to_numeric(X[time_col]).to_numpy(), kind="mergesort")
    X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    rand = outer_splits(y, N_SPLITS, seed)
    fwd = forward_chaining_splits(X[time_col].to_numpy(), n_splits=N_SPLITS)

    def mk():
        return Tab(ceiling_s=budget, seed=seed, n_jobs=cpus)

    def mk_time():
        return Tab(ceiling_s=budget, seed=seed, n_jobs=cpus, time_col=time_col)

    s_rand = _skill(mk, X, y, rand, classes)
    s_fwd = _skill(mk_time, X, y, fwd, classes)
    m, lo, hi = bootstrap_mean_ci(np.array(s_rand) - np.mean(s_fwd))
    return {
        "skill_random": s_rand,
        "skill_forward": s_fwd,
        "optimism_gap": {
            "estimate": float(np.mean(s_rand) - np.mean(s_fwd)),
            "ci_low": lo,
            "ci_high": hi,
        },
    }


def synthetic_drift(
    n: int = 3000, onset: float = 0.6, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Label rule flips sign on x2 after `onset` (fraction of time): random folds see both regimes,
    forward chaining must predict the new regime from the old one."""
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, 1, n))
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    after = t >= onset
    logit = 1.5 * x1 + np.where(after, -2.0, 2.0) * x2
    y = np.where(logit + rng.normal(scale=0.5, size=n) > 0, "a", "b")
    return pd.DataFrame({"t": t, "x1": x1, "x2": x2}), pd.Series(y)


def run(cpus: int = 1, budget: float = 60.0, include_electricity: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    Xs, ys = synthetic_drift()
    syn = compare(Xs, ys, "t", min(budget, 20.0), cpus)
    syn["onset"] = 0.6
    out["synthetic_drift"] = syn
    if include_electricity:
        try:
            ds = datasets.load("electricity")
            chk = check_ordering(ds.X, "date")
            out["electricity_ordering"] = chk
            if chk["usable"]:
                out["electricity"] = compare(ds.X, ds.y, "date", budget, cpus)
        except Exception as e:  # network or data problem: reported, not hidden
            out["electricity_ordering"] = {
                "usable": False,
                "reason": f"could not load: {type(e).__name__}: {e}",
            }
    g = out["synthetic_drift"]["optimism_gap"]
    stmt = f"Synthetic drift: random folds overstate skill by {g['estimate']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] vs forward chaining."
    if "electricity" in out:
        e = out["electricity"]["optimism_gap"]
        stmt += f" Electricity: optimism gap {e['estimate']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]."
    else:
        stmt += f" Electricity not used: {out.get('electricity_ordering', {}).get('reason', 'skipped')}."
    out["statement"] = stmt
    return out
