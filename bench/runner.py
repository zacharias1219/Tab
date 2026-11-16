"""The runner (ADR-003): one subprocess per (system, dataset, fold, seed), measured from outside.

The runner, not the system, measures wall-clock (spawn to exit, refit included) and peak RSS
summed over the whole process tree. It validates the files the system wrote, then evalkit scores
them against labels the system never saw.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from bench import datasets, paths
from evalkit import metrics as ek_metrics
from evalkit.schema import SchemaError, validate, validate_def
from evalkit.score import score_predictions

SYSTEMS: dict[str, str] = {
    "prior": "bench.systems.prior",
    "logreg": "bench.systems.logreg",
    "hgb_default": "bench.systems.hgb_default",
    "tab": "bench.systems.tab",
    "tab_fixed": "bench.systems.tab_fixed",
    "tab_nostack": "bench.systems.tab_nostack",
    "flaml": "bench.systems.flaml",
    "autogluon": "bench.systems.autogluon",
}
TIMEOUT_FACTOR = 1.5
NFR1_FACTOR = 1.10
POLL_S = 0.05
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass
class Measured:
    returncode: int | None
    wall_s: float
    peak_rss_bytes: int
    status: str  # ok | error | timeout | oom


def _tree_rss(proc: psutil.Process) -> int:
    total = 0
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def _kill_tree(proc: psutil.Process) -> None:
    try:
        kids = proc.children(recursive=True)
    except psutil.Error:
        kids = []
    for p in [*kids, proc]:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs([*kids, proc], timeout=10)


def measure_tree(
    args: list[str],
    timeout_s: float,
    mem_limit_bytes: int | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
) -> Measured:
    """Spawn `args` (an argument list, never a shell string) and watch its whole process tree."""
    t0 = time.perf_counter()
    popen = subprocess.Popen(args, env=env, cwd=cwd, stdout=stdout, stderr=stderr)  # noqa: S603
    proc = psutil.Process(popen.pid)
    peak = 0
    status = "ok"
    while True:
        rc = popen.poll()
        if rc is not None:
            break
        peak = max(peak, _tree_rss(proc))
        elapsed = time.perf_counter() - t0
        if elapsed > timeout_s:
            _kill_tree(proc)
            status = "timeout"
            break
        if mem_limit_bytes is not None and peak > mem_limit_bytes:
            _kill_tree(proc)
            status = "oom"
            break
        time.sleep(POLL_S)
    popen.wait()
    wall = time.perf_counter() - t0
    rc = popen.returncode
    if status == "ok" and rc != 0:
        status = "error"
    return Measured(rc, wall, peak, status)


def run_dir(runs_root: Path, system: str, dataset: str, fold: int, seed: int) -> Path:
    if not paths.SYSTEM_RE.match(system):
        raise ValueError(f"invalid system name {system!r}")
    paths.check_name(dataset, "dataset")
    return paths.safe_join(Path(runs_root), system, dataset, f"f{int(fold)}_s{int(seed)}")


def build_command(
    system: str,
    registry: dict[str, str],
    split_dir: Path,
    budget_s: float,
    seed: int,
    cpus: int,
    out: Path,
) -> list[str]:
    if not paths.SYSTEM_RE.match(system):
        raise ValueError(f"invalid system name {system!r}")
    if system not in registry:
        raise ValueError(f"system {system!r} not registered")
    return [
        sys.executable,
        "-m",
        registry[system],
        "--split-dir",
        str(split_dir),
        "--budget-s",
        repr(float(budget_s)),
        "--seed",
        str(int(seed)),
        "--cpus",
        str(int(cpus)),
        "--out",
        str(out),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tail(path: Path, n: int = 1500) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-n:].strip()


def _validate_outputs(
    rd: Path, system: str, n_test: int, budget_s: float, seed: int
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    meta_path, proba_path = rd / "meta.json", rd / "proba.npy"
    if not meta_path.exists() or not proba_path.exists():
        raise ValueError("system did not write proba.npy and meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        validate(meta, "predictions_meta")
    except (SchemaError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid meta.json: {e}") from e
    if meta["system"] != system or meta["seed"] != seed or abs(meta["budget_s"] - budget_s) > 1e-9:
        raise ValueError("meta.json does not match the run (system, seed or budget)")
    proba = np.load(proba_path, allow_pickle=False)
    if proba.dtype != np.float64:
        raise ValueError(f"proba dtype {proba.dtype}, expected float64")
    if proba.shape != (n_test, len(meta["classes"])) or meta["n_rows"] != n_test:
        raise ValueError(
            f"proba shape {proba.shape} does not match {n_test} test rows x {len(meta['classes'])} classes"
        )
    try:
        ek_metrics.check_proba(proba, tol=1e-6)
    except ValueError as e:
        raise ValueError(f"invalid probabilities: {e} (rows must sum to 1)") from e
    log: dict[str, Any] = {}
    if (rd / "system_log.json").exists():
        log = json.loads((rd / "system_log.json").read_text(encoding="utf-8"))
        validate(log, "system_log")
    return proba, list(meta["classes"]), log


def _load_record(rd: Path) -> dict[str, Any] | None:
    p = rd / "record.json"
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        validate_def(rec, "results", "run")
        return rec
    except Exception:
        return None


def run_one(
    system: str,
    dataset: str,
    task_id: int | None,
    fold: int,
    seed: int,
    budget_s: float,
    cpus: int,
    split: datasets.SplitPaths,
    runs_root: Path = paths.RUNS,
    registry: dict[str, str] | None = None,
    mem_limit_gb: float | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    registry = registry or SYSTEMS
    rd = run_dir(runs_root, system, dataset, fold, seed)
    cmd = build_command(system, registry, split.split_dir, budget_s, seed, cpus, rd)
    if resume:
        rec = _load_record(rd)
        if rec is not None:
            return rec
    if rd.exists():
        shutil.rmtree(rd)  # a partial run without a valid record is redone (ADR-004)
    rd.mkdir(parents=True)
    env = dict(os.environ)
    for v in THREAD_VARS:
        env[v] = str(cpus)
    env["PYTHONPATH"] = os.pathsep.join([str(paths.ROOT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    with open(rd / "stdout.log", "wb") as out, open(rd / "stderr.log", "wb") as err:
        m = measure_tree(
            cmd,
            timeout_s=TIMEOUT_FACTOR * budget_s,
            mem_limit_bytes=int(mem_limit_gb * 2**30) if mem_limit_gb else None,
            env=env,
            cwd=paths.ROOT,
            stdout=out,
            stderr=err,
        )
    rec: dict[str, Any] = {
        "run_id": f"{system}__{dataset}__f{fold}__s{seed}",
        "system": system,
        "dataset": dataset,
        "task_id": task_id,
        "fold": int(fold),
        "seed": int(seed),
        "budget_s": float(budget_s),
        "actual_wall_s": max(m.wall_s, 1e-6),
        "peak_rss_gb_tree": m.peak_rss_bytes / 2**30,
        "n_cpus": int(cpus),
        "status": m.status,
        "error": None,
        "exit_code": m.returncode,
        "proba_sha256": None,
        "budget_ratio": max(m.wall_s, 1e-6) / budget_s,
        "within_nfr1": bool(m.wall_s <= NFR1_FACTOR * budget_s),
        "stop_reason": None,
        "n_trials": None,
        "scores": None,
    }
    if m.status == "timeout":
        rec["error"] = (
            f"killed after {TIMEOUT_FACTOR} x ceiling ({TIMEOUT_FACTOR * budget_s:.1f} s)"
        )
    elif m.status == "oom":
        rec["error"] = f"killed: process tree exceeded {mem_limit_gb} GB"
    elif m.status == "error":
        rec["error"] = (
            f"exit code {m.returncode}: {_tail(rd / 'stderr.log')}" or f"exit code {m.returncode}"
        )
    else:
        try:
            proba, classes, log = _validate_outputs(rd, system, split.n_test, budget_s, seed)
            y_test = datasets.read_labels(split)
            y_train = (
                pd.read_parquet(Path(split.split_dir) / "train.parquet", columns=[datasets.TARGET])[
                    datasets.TARGET
                ]
                .astype(str)
                .to_numpy()
            )
            rec["scores"] = score_predictions(
                proba, classes, y_test, y_train, datasets.read_classes(split)
            )
            rec["proba_sha256"] = _sha256(rd / "proba.npy")
            rec["stop_reason"] = log.get("stop_reason")
            rec["n_trials"] = log.get("n_trials")
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"[:2000]
            rec["scores"] = None
    (rd / "record.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec
