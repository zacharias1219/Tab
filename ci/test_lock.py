"""Locked tests (build-loop rule): test files may only change in commits whose message starts with
[test-change]. Checks every commit in the range given (default: origin/main..HEAD)."""

import subprocess
import sys


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout  # noqa: S603,S607


def main() -> int:
    rng = sys.argv[1] if len(sys.argv) > 1 else "origin/main..HEAD"
    try:
        commits = git("rev-list", "--reverse", rng).split()
    except subprocess.CalledProcessError:
        print(f"NOTICE: range {rng} not available; skipping")
        return 0
    bad = []
    for c in commits:
        msg = git("log", "-1", "--format=%s", c)
        files = git("diff-tree", "--no-commit-id", "--name-status", "-r", c).splitlines()
        # modifying or deleting an existing test file is the protected action; adding new tests is fine
        changed = [
            f.split("\t", 1)[1]
            for f in files
            if f[:1] in {"M", "D"} and f.split("\t", 1)[1].startswith("tests/")
        ]
        if changed and not msg.startswith("[test-change]"):
            bad.append((c[:10], msg.strip(), changed))
    for c, m, ch in bad:
        print(f"FAIL {c} '{m}' modified locked tests without [test-change]: {ch}")
    print("test lock OK" if not bad else "test lock FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
