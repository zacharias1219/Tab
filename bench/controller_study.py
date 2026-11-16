"""Budget-controller study (PREREG section 6). Tab fixed 300 s vs Tab + controller, tier 2 x 5 folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench import assemble as asm
from bench import datasets, paths, runner
from evalkit.stats import bootstrap_mean_ci

BUDGET = 300.0
MIN_SAVED = 0.15


def run(
    dsets: list[str],
    folds: list[int],
    cpus: int,
    runs_root: Path,
    seed: int = 0,
    budget: float = BUDGET,
) -> dict[str, Any]:
    root = Path(runs_root).parent / f"{Path(runs_root).name}_controller_{int(budget)}s"
    for d in dsets:
        ds = datasets.load(d)
        for f in folds:
            sp = datasets.write_split(ds, f)
            for s in ("tab_fixed", "tab"):
                runner.run_one(s, ds.name, ds.task_id, f, seed, budget, cpus, sp, root)
    runs = asm.collect(root, ["tab_fixed", "tab"], dsets, folds, seed, allow_missing=False)
    return summarize(runs, margin=load_margin())


def load_margin() -> float | None:
    p = paths.ROOT / "prereg" / "tuned_params.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["non_inferiority_margin"])
    return None


def summarize(runs: list[dict[str, Any]], margin: float | None) -> dict[str, Any]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in runs:
        by.setdefault((r["system"], r["dataset"]), []).append(r)
    dsets = sorted({d for (_, d) in by})
    saved, dscore, rows = [], [], []
    for d in dsets:
        fx = sorted(by.get(("tab_fixed", d), []), key=lambda r: r["fold"])
        ct = sorted(by.get(("tab", d), []), key=lambda r: r["fold"])
        if not fx or not ct:
            continue
        t_fx = np.mean([r["actual_wall_s"] for r in fx])
        t_ct = np.mean([r["actual_wall_s"] for r in ct])
        s_fx = np.mean([asm._skill(r) for r in fx])
        s_ct = np.mean([asm._skill(r) for r in ct])
        saved.append(1 - t_ct / t_fx)
        dscore.append(s_ct - s_fx)
        rows.append(
            {
                "dataset": d,
                "compute_saved": float(1 - t_ct / t_fx),
                "score_change": float(s_ct - s_fx),
                "stop_reasons": [r["stop_reason"] for r in ct],
            }
        )
    if not rows:
        return {"statement": "Controller study: no paired data.", "rows": []}
    ms, ls, hs = bootstrap_mean_ci(np.array(saved))
    md, ld, hd = bootstrap_mean_ci(np.array(dscore))
    compute_ok = ms >= MIN_SAVED and ls > 0
    score_ok = margin is not None and ld > -margin
    if compute_ok and score_ok:
        verdict = "success"
    elif compute_ok or score_ok:
        verdict = "partial"
    else:
        verdict = "null"
    stmt = (
        f"Controller saved {ms:.1%} compute [95% CI {ls:.1%}, {hs:.1%}] with score change {md:+.4f} [{ld:+.4f}, {hd:+.4f}] "
        f"over {len(rows)} datasets; non-inferiority margin {margin if margin is not None else 'not set'}. Pre-registered verdict: {verdict}."
    )
    return {
        "statement": stmt,
        "verdict": verdict,
        "compute_saved": {"estimate": ms, "ci_low": ls, "ci_high": hs},
        "score_change": {"estimate": md, "ci_low": ld, "ci_high": hd},
        "margin": margin,
        "min_saved": MIN_SAVED,
        "rows": rows,
    }
