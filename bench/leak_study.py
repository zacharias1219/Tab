"""Full injected-leak study (PREREG section 7): 12 variants x frozen datasets, one seed, plus
null-column false-positive rate and a scan of the unmodified datasets for manual review."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from bench import datasets
from bench.leak_inject import FLIP_SEVERITIES, SEVERITIES, SIMPLE_SET, inject, is_detected
from evalkit.stats import wilson_interval
from tab.leak import DEFAULT_THRESHOLD, detect_leaks

SEED = 20261116
KILL_RECALL = 0.70


def run(
    dataset_names: list[str],
    fold: int = 0,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = SEED,
    loader=datasets.load,
) -> dict[str, Any]:
    cases = []
    fpr = []
    review = []
    for name in dataset_names:
        ds = loader(name)
        tr, te = ds.folds[fold]
        Xtr, ytr = ds.X.iloc[tr].reset_index(drop=True), ds.y.iloc[tr].reset_index(drop=True)
        Xte, yte = ds.X.iloc[te].reset_index(drop=True), ds.y.iloc[te].reset_index(drop=True)
        clean = detect_leaks(Xtr, ytr, X_test=Xte, threshold=threshold, null_columns=5, seed=seed)
        fpr.append({"dataset": name, "null_fpr": clean.null_fpr, "null_n": clean.null_n})
        flagged = [f for f in clean.flags if f["flagged"]]
        if flagged:
            review.append({"dataset": name, "flags": flagged})
        for ty, sevs in SEVERITIES.items():
            for sev in sevs + (FLIP_SEVERITIES if ty == "T2" else []):
                X2, y2, Xt2, exp = inject(Xtr, ytr, Xte, yte, ty, sev, seed)
                rep = detect_leaks(
                    X2, y2, X_test=Xt2, threshold=threshold, null_columns=0, seed=seed
                )
                cases.append(
                    {
                        "dataset": name,
                        "type": ty,
                        "severity": sev,
                        "detected": bool(is_detected(rep, exp)),
                    }
                )
    by = defaultdict(list)
    for c in cases:
        by[(c["type"], c["severity"])].append(c["detected"])
    recall = []
    for (ty, sev), v in sorted(by.items()):
        lo, hi = wilson_interval(int(sum(v)), len(v))
        recall.append(
            {
                "type": ty,
                "severity": sev,
                "n": len(v),
                "recall": float(np.mean(v)),
                "ci_low": lo,
                "ci_high": hi,
                "simple_set": (ty, sev) in SIMPLE_SET,
                "reported_beside": sev in FLIP_SEVERITIES,
            }
        )
    simple = [c["detected"] for c in cases if (c["type"], c["severity"]) in SIMPLE_SET]
    k, n = int(sum(simple)), len(simple)
    lo, hi = wilson_interval(k, n)
    pooled = k / n if n else 0.0
    total_null = sum(f["null_n"] for f in fpr)
    total_fp = sum(round(f["null_fpr"] * f["null_n"]) for f in fpr)
    killed = pooled < KILL_RECALL
    statement = (
        f"Pooled simple-set recall {pooled:.2f} [95% Wilson {lo:.2f}, {hi:.2f}] over {n} cases; "
        f"null-column false-positive rate {total_fp}/{total_null}. "
        + (
            "Kill criterion triggered: the detector is cut to a documented heuristic."
            if killed
            else "Kill criterion (70%) not triggered."
        )
    )
    return {
        "statement": statement,
        "threshold": threshold,
        "n_datasets": len(dataset_names),
        "n_cases": len([c for c in cases if c["severity"] not in FLIP_SEVERITIES]),
        "recall": recall,
        "simple_set_recall": {
            "k": k,
            "n": n,
            "recall": pooled,
            "ci_low": lo,
            "ci_high": hi,
            "kill_triggered": killed,
        },
        "null_fpr": {"false_positives": total_fp, "n_null_columns": total_null, "per_dataset": fpr},
        "manual_review": review,
        "cases": cases,
    }
