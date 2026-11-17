"""NFR-4 / NFR-7 timing gates: run a command and fail if it exceeds a limit in seconds.

python ci/timing_gate.py 180 -- python -m pytest -q
"""

import subprocess
import sys
import time


def main() -> int:
    limit = float(sys.argv[1])
    cmd = sys.argv[sys.argv.index("--") + 1 :]
    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode  # noqa: S603
    took = time.perf_counter() - t0
    print(f"timing gate: {took:.1f} s (limit {limit:.0f} s), exit {rc}")
    if rc != 0:
        return rc
    if took > limit:
        print("FAIL: over the time limit")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
