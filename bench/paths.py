"""Filesystem layout and path safety (threat model: a crafted name must not escape a root)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SPLITS = ROOT / ".cache" / "splits"
LABELS = ROOT / ".cache" / "labels"
OPENML_CACHE = ROOT / ".cache" / "openml"
RESULTS = ROOT / "results"
SELECTED = ROOT / "bench" / "selected_datasets.json"
SELECTED_SHA = ROOT / "bench" / "selected_datasets.sha256"

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SYSTEM_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def check_name(name: str, what: str = "name") -> str:
    if not NAME_RE.match(name) or name in {".", ".."} or ".." in name:
        raise ValueError(f"invalid {what}: {name!r}")
    return name


def safe_join(root: Path, *parts: str) -> Path:
    root = Path(root).resolve()
    p = root.joinpath(*parts).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"path {p} escapes {root}")
    return p
