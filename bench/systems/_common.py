"""Shared CLI for system wrappers (ARCHITECTURE 4.2).

A system sees train.parquet (features + __y) and test_X.parquet, writes proba.npy, meta.json and
optionally system_log.json. It never sees test labels and never computes a score (HF-3).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

TARGET = "__y"


@dataclass
class Context:
    budget_s: float
    seed: int
    cpus: int
    out: Path
    process_start: float
    extra_meta: dict[str, Any] = field(default_factory=dict)

    def remaining_s(self, margin_s: float = 0.5, margin_frac: float = 0.03) -> float:
        """Budget left for fitting: ceiling minus time already spent since process start
        (interpreter start-up and imports count) minus a margin for writing outputs."""
        spent = time.time() - self.process_start
        return max(self.budget_s - spent - margin_s - margin_frac * self.budget_s, 0.5)


FitPredict = Callable[
    [pd.DataFrame, pd.Series, pd.DataFrame, Context], tuple[np.ndarray, list[str], dict[str, Any]]
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--budget-s", required=True, type=float)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--cpus", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    return ap.parse_args(argv)


def run_system(
    name: str, version: str, fit_predict: FitPredict, argv: list[str] | None = None
) -> None:
    start = psutil.Process().create_time()
    a = parse_args(argv)
    train = pd.read_parquet(a.split_dir / "train.parquet")
    test = pd.read_parquet(a.split_dir / "test_X.parquet")
    y = train.pop(TARGET).astype(str)
    ctx = Context(a.budget_s, a.seed, a.cpus, a.out, start)
    proba, classes, log = fit_predict(train, y, test, ctx)
    proba = np.ascontiguousarray(proba, dtype=np.float64)
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "proba.npy", proba, allow_pickle=False)
    meta = {
        "schema_version": "1",
        "system": name,
        "system_version": version,
        "classes": [str(c) for c in classes],
        "n_rows": int(proba.shape[0]),
        "seed": a.seed,
        "budget_s": a.budget_s,
        "dtype": "float64",
        **ctx.extra_meta,
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if log:
        (a.out / "system_log.json").write_text(
            json.dumps(log, indent=2, default=str), encoding="utf-8"
        )
