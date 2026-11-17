"""Spawns a grandchild that allocates MB megabytes, so the runner must watch the whole tree."""
import subprocess
import sys

mb = int(sys.argv[sys.argv.index("--mb") + 1]) if "--mb" in sys.argv else 0
code = (
    "import numpy as np, time\n"
    f"a = np.ones({mb} * 1024 * 1024 // 8)\n"
    "time.sleep(1.5)\n"
)
subprocess.run([sys.executable, "-c", code], check=True)
