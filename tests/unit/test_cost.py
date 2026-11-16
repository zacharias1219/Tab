"""US-12: peak memory across the whole process tree, measured within 15%."""

import sys

from bench.runner import measure_tree

MB = 500


def test_tree_peak_memory_within_15_percent():
    base = measure_tree([sys.executable, "-m", "tests.fake_systems.allocator", "--mb", "0"], timeout_s=60)
    big = measure_tree([sys.executable, "-m", "tests.fake_systems.allocator", "--mb", str(MB)], timeout_s=60)
    assert base.returncode == 0 and big.returncode == 0
    delta_mb = (big.peak_rss_bytes - base.peak_rss_bytes) / 2**20
    assert abs(delta_mb - MB) / MB <= 0.15, delta_mb
    assert big.wall_s >= 1.5


def test_wall_clock_measured_from_outside():
    r = measure_tree([sys.executable, "-c", "import time; time.sleep(0.6)"], timeout_s=30)
    assert 0.6 <= r.wall_s < 5
