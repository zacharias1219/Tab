"""Assemble results.json from run records (ADR-004). Validates against the schema before writing."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import psutil

import evalkit
from bench import datasets, paths, runner
from evalkit.headline import build_headline, compare_systems
from evalkit.schema import validate

PRIMARY = "tab"
NOT_BASELINES = {
    "prior",
    "tab_fixed",
    "tab_nostack",
}  # stub and study arms are not headline baselines
NFR3_LIMIT_GB = 8.0


def git_sha(root: Path = paths.ROOT) -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()  # noqa: S607
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()  # noqa: S607
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "0000000"


def hardware() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine() or "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu_model": cpu,
        "n_logical_cpus": psutil.cpu_count(logical=True) or 1,
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def versions() -> dict[str, str]:
    import tab

    out = {"tab": tab.__version__, "evalkit": evalkit.__version__}
    for pkg in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pyarrow",
        "flaml",
        "autogluon.tabular",
    ):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def dataset_list_sha() -> str:
    return datasets.file_sha256(paths.SELECTED)


def collect(
    runs_root: Path,
    systems: list[str],
    dsets: list[str],
    folds: list[int],
    seed: int,
    allow_missing: bool,
) -> list[dict[str, Any]]:
    out, missing = [], []
    for s in systems:
        for d in dsets:
            for f in folds:
                p = runner.run_dir(runs_root, s, d, f, seed) / "record.json"
                if p.exists():
                    out.append(json.loads(p.read_text(encoding="utf-8")))
                else:
                    missing.append(f"{s}/{d}/f{f}")
    if missing and not allow_missing:
        raise ValueError(
            f"{len(missing)} expected runs have no record (no run may be dropped): {missing[:10]}"
        )
    return out


def _skill(rec: dict[str, Any]) -> float:
    """PREREG section 1: a failed run scores as the class-prior predictor (S = 0)."""
    return float(rec["scores"]["skill"]) if rec["status"] == "ok" else 0.0


def per_dataset_table(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    tab: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[(r["system"], r["dataset"])].append(r)
    for (s, d), rs in grouped.items():
        rs = sorted(rs, key=lambda r: r["fold"])
        ok = [r for r in rs if r["status"] == "ok"]
        tab[s][d] = {
            "folds": [r["fold"] for r in rs],
            "skill": np.array([_skill(r) for r in rs]),
            "auc": np.array([r["scores"]["auc"] for r in ok if r["scores"]["auc"] is not None]),
            "log_loss": [r["scores"]["log_loss"] if r["status"] == "ok" else None for r in rs],
            "actual_s": [r["actual_wall_s"] for r in rs],
            "rss": [r["peak_rss_gb_tree"] for r in rs],
            "status": [r["status"] for r in rs],
        }
    return tab


def analyses(
    runs: list[dict[str, Any]], primary: str = PRIMARY
) -> tuple[dict[str, Any], dict[str, Any]]:
    table = per_dataset_table(runs)
    systems = sorted(table)
    per_dataset = []
    for s in systems:
        for d, v in sorted(table[s].items()):
            row = {
                "system": s,
                "dataset": d,
                "n_folds": len(v["folds"]),
                "mean_skill": float(np.mean(v["skill"])),
                "mean_log_loss": float(np.mean([x for x in v["log_loss"] if x is not None]))
                if any(x is not None for x in v["log_loss"])
                else None,
                "mean_auc": float(np.mean(v["auc"])) if v["auc"].size else None,
                "median_actual_s": float(np.median(v["actual_s"])),
                "max_peak_rss_gb": float(np.max(v["rss"])),
                "failures": [st for st in v["status"] if st != "ok"],
            }
            if s != primary and primary in table and d in table[primary]:
                a, b = table[primary][d]["skill"], v["skill"]
                n = min(a.size, b.size)
                from evalkit.stats import bootstrap_mean_ci, verdict

                diff = a[:n] - b[:n]
                m, lo, hi = bootstrap_mean_ci(diff)
                row["vs_primary"] = {
                    "primary": primary,
                    "mean_diff": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "verdict": verdict(diff) if n > 1 else "insufficient",
                }
            per_dataset.append(row)

    comparisons = []
    if primary in table:
        for s in systems:
            if s == primary or s in NOT_BASELINES:
                continue
            skill_a = {d: v["skill"] for d, v in table[primary].items()}
            skill_b = {d: v["skill"] for d, v in table[s].items()}
            auc_a = {d: v["auc"] for d, v in table[primary].items() if v["auc"].size}
            auc_b = {d: v["auc"] for d, v in table[s].items() if v["auc"].size}
            comparisons.append(compare_systems(s, skill_a, skill_b, auc_a, auc_b))

    prim_runs = [r for r in runs if r["system"] == primary]
    n_within = sum(r["within_nfr1"] for r in prim_runs)
    nfr1 = {
        "system": primary,
        "n_runs": len(prim_runs),
        "n_within": int(n_within),
        "fraction_within": (n_within / len(prim_runs)) if prim_runs else None,
        "passes": (n_within / len(prim_runs) >= 0.95) if prim_runs else None,
        "overruns": [
            {
                "run_id": r["run_id"],
                "budget_ratio": r["budget_ratio"],
                "actual_wall_s": r["actual_wall_s"],
            }
            for r in prim_runs
            if not r["within_nfr1"]
        ],
    }

    cost = []
    for s in systems:
        rs = [r for r in runs if r["system"] == s]
        cost.append(
            {
                "system": s,
                "n_runs": len(rs),
                "median_actual_s": float(np.median([r["actual_wall_s"] for r in rs])),
                "p95_actual_s": float(np.percentile([r["actual_wall_s"] for r in rs], 95)),
                "max_peak_rss_gb": float(max(r["peak_rss_gb_tree"] for r in rs)),
                "median_budget_ratio": float(np.median([r["budget_ratio"] for r in rs])),
                "n_failed": sum(r["status"] != "ok" for r in rs),
                "statuses": {
                    st: sum(r["status"] == st for r in rs)
                    for st in ("ok", "error", "timeout", "oom")
                },
            }
        )

    out: dict[str, Any] = {"per_dataset": per_dataset, "nfr1": nfr1, "cost": cost}
    floor = next((c for c in comparisons if c["baseline"] == "hgb_default"), None)
    if floor is not None:
        paid = floor["verdict"] == "win"
        out["floor_check"] = {
            "baseline": "hgb_default",
            "verdict": floor["verdict"],
            "passes": paid,
            "statement": "Tuning paid: Tab beats default HistGradientBoosting (interval excludes zero)."
            if paid
            else "Tuning did not pay: Tab does not beat default HistGradientBoosting with an interval excluding zero.",
        }
    if prim_runs:
        peak = max(prim_runs, key=lambda r: r["peak_rss_gb_tree"])
        out["memory"] = {
            "system": primary,
            "max_peak_rss_gb": peak["peak_rss_gb_tree"],
            "dataset": peak["dataset"],
            "limit_gb": NFR3_LIMIT_GB,
            "passes": peak["peak_rss_gb_tree"] <= NFR3_LIMIT_GB,
        }
    n_datasets = len(table.get(primary, {}))
    headline = build_headline(primary, comparisons, n_datasets)
    return out, headline


def assemble(
    runs_root: Path,
    out_path: Path,
    mode: str,
    budget_s: float,
    systems: list[str],
    dsets: list[str],
    folds: list[int],
    seed: int = 0,
    cpus: int = 1,
    allow_missing: bool = False,
    studies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from tab import config as tab_config

    runs = collect(runs_root, systems, dsets, folds, seed, allow_missing)
    if not runs:
        raise ValueError("no run records found")
    an, headline = analyses(runs)
    for k, v in (studies or {}).items():
        an[k] = v
    doc = {
        "schema_version": "1",
        "project": "3A-tab",
        "git_sha": git_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hardware(),
        "versions": versions(),
        "dataset_list_sha256": dataset_list_sha(),
        "config": {
            "mode": mode,
            "budget_s": float(budget_s),
            "folds": list(folds),
            "systems": list(systems),
            "datasets": list(dsets),
            "n_cpus": int(cpus),
            "provisional_params": bool(
                tab_config.PROVISIONAL
                and not (paths.ROOT / "prereg" / "tuned_params.json").exists()
            ),
        },
        "runs": runs,
        "analyses": an,
        "headline": headline,
    }
    doc = json.loads(json.dumps(doc, default=_json_default))
    validate(doc, "results")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))
