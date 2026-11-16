"""Stacking ablation (PREREG section 9): Tab vs Tab without stacking, tier 1, paired."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bench import assemble as asm
from bench import datasets, runner
from evalkit.headline import compare_systems


def run(
    dsets: list[str], folds: list[int], budget: float, cpus: int, runs_root: Path, seed: int = 0
) -> dict[str, Any]:
    for d in dsets:
        ds = datasets.load(d)
        for f in folds:
            sp = datasets.write_split(ds, f)
            for s in ("tab", "tab_nostack"):
                runner.run_one(s, ds.name, ds.task_id, f, seed, budget, cpus, sp, runs_root)
    return summarize(
        asm.collect(runs_root, ["tab", "tab_nostack"], dsets, folds, seed, allow_missing=False)
    )


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    table = asm.per_dataset_table(runs)
    comp = compare_systems(
        "tab_nostack",
        {d: v["skill"] for d, v in table.get("tab", {}).items()},
        {d: v["skill"] for d, v in table.get("tab_nostack", {}).items()},
    )
    word = {
        "win": "helps",
        "loss": "hurts",
        "tie": "makes no detectable difference",
        "insufficient": "has insufficient data",
    }[comp["verdict"]]
    md = comp["mean_diff"]
    stmt = f"Stacking {word}" + (
        f": mean skill difference {md:+.4f} [{comp['ci_low']:+.4f}, {comp['ci_high']:+.4f}]."
        if md is not None
        else "."
    )
    return {"statement": stmt, "comparison": comp}
