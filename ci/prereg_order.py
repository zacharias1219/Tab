"""HF-6: no benchmark result may exist in git history before the pre-registration commit.

Benchmark results are any committed file under results/full/ or results/studies/, and
prereg/tuned_params.json must also precede them (PREREG section 5)."""

from __future__ import annotations

import subprocess
import sys


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603,S607


def first_commit_touching(path: str) -> str | None:
    out = git("log", "--reverse", "--format=%H", "--", path)
    return out.splitlines()[0] if out else None


def is_ancestor(a: str, b: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b]).returncode == 0  # noqa: S603,S607


def main() -> int:
    prereg = first_commit_touching("prereg/PREREG.md")
    if prereg is None:
        print("FAIL: prereg/PREREG.md was never committed")
        return 1
    result = None
    for path in ("results/full", "results/studies"):
        c = first_commit_touching(path)
        if c and (result is None or is_ancestor(c, result)):
            result = c
    if result is None:
        print(f"OK: PREREG.md committed at {prereg[:10]}; no benchmark results committed yet")
        return 0
    problems = []
    if not (is_ancestor(prereg, result) and prereg != result):
        problems.append(
            f"first result commit {result[:10]} does not come after PREREG.md commit {prereg[:10]}"
        )
    tuned = first_commit_touching("prereg/tuned_params.json")
    if tuned is None or not (is_ancestor(tuned, result) and tuned != result):
        problems.append("prereg/tuned_params.json must be committed before the first result")
    for p in problems:
        print(f"FAIL (HF-6): {p}")
    if not problems:
        print(
            f"OK: PREREG.md {prereg[:10]} and tuned_params {tuned[:10]} precede first result {result[:10]}"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
