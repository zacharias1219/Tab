"""Set every tuned parameter on the development pool only (PREREG section 5), then write
prereg/tuned_params.json. Run once, commit the file, then run the benchmark.

    python -m bench.tune_dev --cpus 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from bench import assemble as asm
from bench import datasets, devpool, paths, runner
from tab.config import ControllerConfig
from tab.controller import simulate
from tab.leak import detect_leaks

BUDGET = 60.0
W_GRID = [4, 6, 8, 12, 16]
DELTA_GRID = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
THRESH_GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
MAX_LOSS_INCREASE = (
    0.002  # mean relative increase in best inner loss allowed when choosing (W, delta)
)
MAX_FLAG_RATE = 0.02  # fraction of original dev columns the detector may flag


def choose_controller(curves: list[list[tuple[float, float]]]) -> dict[str, Any]:
    """curves: per run, list of (loss, wall_s) in trial order (controller off). Pick (W, delta) with
    the largest mean compute saved subject to mean relative best-loss increase <= MAX_LOSS_INCREASE;
    ties go to the larger W (more conservative)."""
    best = None
    table = []
    for w, d in itertools.product(W_GRID, DELTA_GRID):
        cfg = ControllerConfig(window=w, delta=d)
        saved, worse = [], []
        for c in curves:
            losses = [x[0] if x[0] is not None else float("nan") for x in c]
            walls = [x[1] for x in c]
            if not losses:
                continue
            total = sum(walls)
            res = simulate(losses, cfg, trial_s=1.0, ceiling_s=float(len(losses)))
            used = sum(walls[: res.n_trials])
            full_best = np.nanmin(losses)
            saved.append(1 - used / total if total > 0 else 0.0)
            worse.append((res.best_loss - full_best) / max(abs(full_best), 1e-12))
        row = {
            "window": w,
            "delta": d,
            "mean_saved": float(np.mean(saved)),
            "mean_loss_increase": float(np.mean(worse)),
        }
        table.append(row)
        if row["mean_loss_increase"] <= MAX_LOSS_INCREASE:
            key = (row["mean_saved"], w)
            if best is None or key > best[0]:
                best = (key, row)
    chosen = best[1] if best else {"window": max(W_GRID), "delta": min(DELTA_GRID)}
    return {"chosen": chosen, "grid": table}


def choose_threshold(rates: dict[float, float]) -> float:
    ok = [t for t in sorted(rates) if rates[t] <= MAX_FLAG_RATE]
    return ok[0] if ok else max(rates)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--runs", default=str(paths.RUNS.parent / "runs_dev"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args(argv)
    sel = datasets.selected_list()
    if sel.get("status") != "frozen" or not datasets.check_frozen():
        print("refusing: freeze the dataset list first")
        return 2
    drawn = [e["task_id"] for e in sel["tier1"] + sel["tier2"]]
    dev = devpool.draw_dev_pool(sel["eligible"], drawn)
    folds = [int(f) for f in a.folds.split(",")]
    root = Path(a.runs)
    names = []
    for tid in dev:
        ds = datasets.load_openml_task(tid, f"task{tid}")
        names.append(ds.name)
        for f in folds:
            sp = datasets.write_split(ds, f)
            runner.run_one("tab_fixed", ds.name, tid, f, 0, BUDGET, a.cpus, sp, root)
    runs = asm.collect(root, ["tab_fixed"], names, folds, 0, allow_missing=False)

    # non-inferiority margin: 0.5 x median over dev datasets of fold-to-fold std of S
    stds = []
    for n in names:
        s = [asm._skill(r) for r in runs if r["dataset"] == n]
        stds.append(float(np.std(s, ddof=1)))
    margin = 0.5 * float(np.median(stds))

    curves, ece_b, ece_a = [], [], []
    for r in runs:
        log = runner.run_dir(root, "tab_fixed", r["dataset"], r["fold"], 0) / "system_log.json"
        if log.exists():
            extra = json.loads(log.read_text(encoding="utf-8")).get("extra", {})
            curves.append([(h["loss"], h["wall_s"]) for h in extra.get("history", [])])
            if extra.get("ece_before") is not None:
                ece_b.append(extra["ece_before"])
                ece_a.append(extra["ece_after"])
    ctrl = choose_controller(curves)
    calibrate = bool(np.mean(ece_a) <= np.mean(ece_b)) if ece_b else True

    rates = {}
    for t in THRESH_GRID:
        flagged = total = 0
        for tid in dev:
            ds = datasets.load_openml_task(tid, f"task{tid}")
            tr, _ = ds.folds[0]
            rep = detect_leaks(ds.X.iloc[tr], ds.y.iloc[tr], threshold=t, null_columns=0)
            col_flags = [f for f in rep.flags if f["column"] is not None]
            flagged += sum(f["flagged"] for f in col_flags)
            total += len(col_flags)
        rates[t] = flagged / max(total, 1)
    thresh = choose_threshold(rates)

    out = {
        "controller_window": ctrl["chosen"]["window"],
        "controller_delta": ctrl["chosen"]["delta"],
        "leak_threshold": thresh,
        "calibrate": calibrate,
        "non_inferiority_margin": margin,
        "provenance": {
            "dev_task_ids": dev,
            "folds": folds,
            "budget_s": BUDGET,
            "git_sha": asm.git_sha(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "controller_grid": ctrl["grid"],
            "threshold_flag_rates": rates,
            "fold_skill_std": dict(zip(names, stds, strict=True)),
            "ece_before_mean": float(np.mean(ece_b)) if ece_b else None,
            "ece_after_mean": float(np.mean(ece_a)) if ece_a else None,
        },
    }
    target = paths.ROOT / "prereg" / "tuned_params.json"
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}. Commit it before any tier-1 result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
