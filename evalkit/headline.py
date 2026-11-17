"""Cross-dataset comparisons and the headline statement (HF-4: skill score only)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evalkit import stats

CLAIM_MIN_DATASETS = 10

PerFold = Mapping[str, NDArray[np.floating]]


def compare_systems(
    baseline: str,
    skill_a: PerFold,
    skill_b: PerFold,
    auc_a: PerFold | None = None,
    auc_b: PerFold | None = None,
) -> dict[str, Any]:
    """Compare system A (Tab) with baseline B on per-dataset, per-fold skill scores.

    Headline: mean over shared datasets of (mean-fold S_A - mean-fold S_B), bootstrap over datasets.
    Per-dataset verdict: bootstrap over paired fold differences. AUC is reported, never decisive.
    """
    shared = sorted(set(skill_a) & set(skill_b))
    wins = ties = losses = 0
    diffs = []
    for d in shared:
        a, b = np.asarray(skill_a[d], float), np.asarray(skill_b[d], float)
        n = min(a.size, b.size)
        fold_diff = a[:n] - b[:n]
        diffs.append(float(fold_diff.mean()))
        # one fold cannot carry a per-dataset verdict: it counts as a tie
        v = stats.verdict(fold_diff) if n >= 2 else "tie"
        wins += v == "win"
        ties += v == "tie"
        losses += v == "loss"
    out: dict[str, Any] = {
        "baseline": baseline,
        "n_datasets": len(shared),
        "mean_diff": None,
        "ci_low": None,
        "ci_high": None,
        "verdict": "insufficient",
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "secondary": {},
    }
    if shared:
        m, lo, hi = stats.bootstrap_mean_ci(np.array(diffs))
        out.update(mean_diff=m, ci_low=lo, ci_high=hi)
        out["verdict"] = "win" if lo > 0 else "loss" if hi < 0 else "tie"
        if len(shared) == 1:
            out["verdict"] = "insufficient"
    if auc_a is not None and auc_b is not None:
        auc_shared = [d for d in shared if d in auc_a and d in auc_b]
        if auc_shared:
            ad = [float(np.mean(auc_a[d]) - np.mean(auc_b[d])) for d in auc_shared]
            out["secondary"] = {
                "auc_mean_diff": float(np.mean(ad)),
                "auc_n_datasets": len(ad),
                "note": "AUC is secondary and cannot change the verdict (HF-4)",
            }
    return out


def _phrase(c: Mapping[str, Any]) -> str:
    v = c["verdict"]
    if v == "insufficient" or c["mean_diff"] is None:
        return f"vs {c['baseline']}: insufficient data (n = {c['n_datasets']})"
    word = {"win": "better", "loss": "worse", "tie": "tie (interval includes zero)"}[v]
    return (
        f"vs {c['baseline']}: {word}, mean skill difference {c['mean_diff']:+.4f} "
        f"[95% CI {c['ci_low']:+.4f}, {c['ci_high']:+.4f}] over {c['n_datasets']} datasets "
        f"({c['wins']} better / {c['ties']} tied / {c['losses']} worse)"
    )


def build_headline(
    system: str, comparisons: list[dict[str, Any]], n_datasets: int
) -> dict[str, Any]:
    is_claim = n_datasets >= CLAIM_MIN_DATASETS and all(
        c["verdict"] != "insufficient" for c in comparisons
    )
    parts = [_phrase(c) for c in comparisons] or ["no baseline compared"]
    prefix = "" if is_claim else f"n = {n_datasets}, not yet a claim. "
    return {
        "metric": "skill_score_log_loss",
        "system": system,
        "is_claim": bool(is_claim),
        "statement": prefix + f"{system} skill score (log loss) " + "; ".join(parts) + ".",
        "comparisons": comparisons,
    }
