"""Dataset loading and split writing (ARCHITECTURE section 5 step 1).

Sources: bundled fixtures (CI, no network) and OpenML tasks (cached, sha256-checked).
A split directory holds train.parquet (features + __y) and test_X.parquet only. Test labels go to
a separate label directory that is never passed to a system.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from bench import paths

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TARGET = "__y"
OPENML_API = "https://www.openml.org/api/v1/json"


@dataclass
class Dataset:
    name: str
    task_id: int | None
    X: pd.DataFrame
    y: pd.Series
    classes: list[str]
    folds: list[tuple[np.ndarray, np.ndarray]]
    source: str


@dataclass
class SplitPaths:
    dataset: str
    fold: int
    split_dir: Path
    label_dir: Path
    n_test: int


# --------------------------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_frozen(selected: Path = paths.SELECTED, sha_file: Path = paths.SELECTED_SHA) -> bool:
    """HF-2: the frozen list must hash to the committed value. Returns False if not frozen yet."""
    if not Path(sha_file).exists():
        return False
    want = Path(sha_file).read_text(encoding="utf-8").split()[0]
    got = file_sha256(Path(selected))
    if want != got:
        raise ValueError(f"HF-2: {selected} changed after freeze (sha256 {got} != {want})")
    return True


def selected_list(selected: Path = paths.SELECTED) -> dict[str, Any]:
    return json.loads(Path(selected).read_text(encoding="utf-8"))


def record_or_check_sha(manifest: Path, key: str, f: Path) -> str:
    manifest = Path(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    h = file_sha256(f)
    if key in data and data[key] != h:
        raise ValueError(
            f"cached file {f} sha256 {h} differs from first download {data[key]} (data drift)"
        )
    data[key] = h
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return h


# --------------------------------------------------------------------------------------------
def _stratified_folds(
    y: pd.Series, n: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y.astype(str))]


def load_fixture(name: str) -> Dataset:
    csv = FIXTURES / f"{name}.csv"
    dtypes = json.loads((FIXTURES / f"{name}.dtypes.json").read_text(encoding="utf-8"))
    df = pd.read_csv(csv, keep_default_na=True)
    for c, t in dtypes.items():
        if t == "category":
            df[c] = df[c].astype("category")
        elif t == "boolean":
            df[c] = df[c].astype("boolean")
        else:
            df[c] = df[c].astype(t)
    y = df.pop(TARGET).astype(str)
    return Dataset(name, None, df, y, sorted(y.unique()), _stratified_folds(y), "fixture")


def parse_split_arff(
    text: str, n_folds: int = 5, repeat: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse an OpenML task split ARFF (type,rowid,repeat,fold) into (train, test) index arrays."""
    in_data = False
    rows: dict[int, dict[str, list[int]]] = {f: {"TRAIN": [], "TEST": []} for f in range(n_folds)}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.upper().startswith("@DATA"):
            in_data = True
            continue
        if not in_data:
            continue
        typ, rowid, rep, fold = [p.strip().strip("'\"") for p in line.split(",")[:4]]
        if int(float(rep)) != repeat:
            continue
        f = int(float(fold))
        if f in rows:
            rows[f][typ.upper()].append(int(float(rowid)))
    return [
        (np.array(sorted(rows[f]["TRAIN"])), np.array(sorted(rows[f]["TEST"])))
        for f in range(n_folds)
    ]


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not url.startswith("https://"):
            raise ValueError("only https downloads are allowed")
        with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - https only, checked above
            dest.write_bytes(r.read())
    return dest


def load_openml_task(
    task_id: int, name: str | None = None, cache: Path = paths.OPENML_CACHE, n_folds: int = 5
) -> Dataset:
    """Load an OpenML task: dataset via sklearn.fetch_openml (cached), splits from the task's ARFF.
    Every downloaded file is sha256-recorded on first download and checked afterwards."""
    from sklearn.datasets import fetch_openml

    tdir = Path(cache) / f"task_{task_id}"
    manifest = Path(cache) / "manifest.json"
    tjson = _fetch(f"{OPENML_API}/task/{task_id}", tdir / "task.json")
    record_or_check_sha(manifest, f"task_{task_id}/task.json", tjson)
    task = json.loads(tjson.read_text(encoding="utf-8"))["task"]
    inputs = {i["name"]: i for i in task["input"]}
    did = int(inputs["source_data"]["data_set"]["data_set_id"])
    target = inputs["source_data"]["data_set"]["target_feature"]
    splits_url = inputs["estimation_procedure"]["estimation_procedure"]["data_splits_url"]
    sfile = _fetch(splits_url, tdir / "splits.arff")
    record_or_check_sha(manifest, f"task_{task_id}/splits.arff", sfile)
    bunch = fetch_openml(
        data_id=did, as_frame=True, data_home=str(Path(cache) / "sklearn"), parser="pandas"
    )
    frame = bunch.frame
    y = frame.pop(target).astype(str)
    X = frame
    pq = tdir / "data.parquet"
    if not pq.exists():
        X.assign(**{TARGET: y}).to_parquet(pq, index=False)
    record_or_check_sha(manifest, f"task_{task_id}/data.parquet", pq)
    folds = parse_split_arff(sfile.read_text(encoding="utf-8"), n_folds=max(n_folds, 10))[:n_folds]
    return Dataset(
        name or bunch.details.get("name", f"task{task_id}"),
        task_id,
        X,
        y,
        sorted(y.unique()),
        folds,
        "openml",
    )


def load(name: str) -> Dataset:
    if name.startswith("fixture_"):
        return load_fixture(name)
    if name.startswith("openml:"):
        return load_openml_task(int(name.split(":", 1)[1]))
    sel = selected_list()
    for tier in ("tier1", "tier2", "dev_pool", "extra"):
        for entry in sel.get(tier, []):
            if entry.get("name") == name:
                return load_openml_task(int(entry["task_id"]), name)
    raise ValueError(f"unknown dataset {name!r}: not a fixture and not in {paths.SELECTED.name}")


# --------------------------------------------------------------------------------------------
def write_split(
    ds: Dataset, fold: int, splits_root: Path = paths.SPLITS, labels_root: Path = paths.LABELS
) -> SplitPaths:
    paths.check_name(ds.name, "dataset")
    tr, te = ds.folds[fold]
    sdir = paths.safe_join(Path(splits_root), ds.name, f"f{fold}")
    ldir = paths.safe_join(Path(labels_root), ds.name, f"f{fold}")
    sdir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    train = ds.X.iloc[tr].reset_index(drop=True).copy()
    train[TARGET] = ds.y.iloc[tr].to_numpy()
    train.to_parquet(sdir / "train.parquet", index=False)
    ds.X.iloc[te].reset_index(drop=True).to_parquet(sdir / "test_X.parquet", index=False)
    (ldir / "test_y.json").write_text(
        json.dumps(ds.y.iloc[te].astype(str).tolist()), encoding="utf-8"
    )
    (ldir / "classes.json").write_text(
        json.dumps(
            {"classes": ds.classes, "dataset": ds.name, "fold": fold, "task_id": ds.task_id}
        ),
        encoding="utf-8",
    )
    return SplitPaths(ds.name, fold, sdir, ldir, len(te))


def read_labels(sp: SplitPaths) -> np.ndarray:
    return np.array(
        json.loads((Path(sp.label_dir) / "test_y.json").read_text(encoding="utf-8")), dtype=object
    )


def read_classes(sp: SplitPaths) -> list[str]:
    return list(
        json.loads((Path(sp.label_dir) / "classes.json").read_text(encoding="utf-8"))["classes"]
    )
