"""`make verify`: recompute every score from runs/ and compare with results.json (threat model:
result tampering). Also re-hashes every proba.npy against its record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bench import datasets, paths, runner
from evalkit.schema import validate
from evalkit.score import score_predictions


def verify(
    results_path: Path,
    runs_root: Path = paths.RUNS,
    splits_root: Path = paths.SPLITS,
    labels_root: Path = paths.LABELS,
) -> list[str]:
    doc = json.loads(Path(results_path).read_text(encoding="utf-8"))
    validate(doc, "results")
    problems: list[str] = []
    for r in doc["runs"]:
        if r["status"] != "ok":
            continue
        rd = runner.run_dir(runs_root, r["system"], r["dataset"], r["fold"], r["seed"])
        proba_path = rd / "proba.npy"
        if not proba_path.exists():
            problems.append(f"{r['run_id']}: proba.npy missing")
            continue
        if hashlib.sha256(proba_path.read_bytes()).hexdigest() != r["proba_sha256"]:
            problems.append(f"{r['run_id']}: proba.npy hash mismatch")
            continue
        meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
        sp = datasets.SplitPaths(
            r["dataset"],
            r["fold"],
            paths.safe_join(splits_root, r["dataset"], f"f{r['fold']}"),
            paths.safe_join(labels_root, r["dataset"], f"f{r['fold']}"),
            0,
        )
        y_test = datasets.read_labels(sp)
        y_train = (
            pd.read_parquet(sp.split_dir / "train.parquet", columns=[datasets.TARGET])[
                datasets.TARGET
            ]
            .astype(str)
            .to_numpy()
        )
        s = score_predictions(
            np.load(proba_path, allow_pickle=False),
            meta["classes"],
            y_test,
            y_train,
            datasets.read_classes(sp),
        )
        for k in ("log_loss", "skill", "ece"):
            if not np.isclose(s[k], r["scores"][k], rtol=1e-9, atol=1e-12):
                problems.append(
                    f"{r['run_id']}: {k} recomputed {s[k]} != recorded {r['scores'][k]}"
                )
    return problems
