"""NFR-6 / HF-7: tab/ imports neither AutoGluon nor FLAML, nor the harness scorer."""

import ast
import subprocess
import sys
from pathlib import Path

import tab

BANNED = {"flaml", "autogluon", "evalkit", "bench"}


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module or ""


def test_static_imports():
    root = Path(tab.__file__).parent
    files = list(root.rglob("*.py"))
    assert files
    for f in files:
        for name in _imports(f):
            assert name.split(".")[0] not in BANNED, f"{f}: imports {name}"


def test_no_dynamic_import_strings():
    root = Path(tab.__file__).parent
    for f in root.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "importlib" not in text and "__import__" not in text, f


def test_runtime_modules():
    code = "import sys, tab; from tab import Tab; print(','.join(sorted(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    mods = {m.split(".")[0] for m in out.strip().split(",")}
    assert not mods & BANNED
