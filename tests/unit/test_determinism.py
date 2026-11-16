"""NFR-2: trial-count mode with a fixed seed gives bitwise-identical predictions on one machine."""

import subprocess
import sys
import textwrap

import numpy as np

from tab import Tab
from tests.conftest import mixed_frame


def test_in_process_bitwise_identical():
    X, y = mixed_frame(300, 5, 3)
    a = Tab(max_trials=5, seed=11, n_jobs=2).fit(X, y).predict_proba(X)
    b = Tab(max_trials=5, seed=11, n_jobs=2).fit(X, y).predict_proba(X)
    assert a.tobytes() == b.tobytes()


SCRIPT = textwrap.dedent(
    """
    import hashlib, sys
    sys.path.insert(0, {root!r})
    from tests.conftest import mixed_frame
    from tab import Tab
    X, y = mixed_frame(300, 5, 2)
    p = Tab(max_trials=5, seed=3).fit(X, y).predict_proba(X)
    print(hashlib.sha256(p.tobytes()).hexdigest())
    """
)


def test_separate_processes_with_different_hash_seeds():
    import os
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    digests = set()
    for hs in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        out = subprocess.run([sys.executable, "-c", SCRIPT.format(root=root)], env=env, capture_output=True, text=True, check=True)
        digests.add(out.stdout.strip())
    assert len(digests) == 1


def test_different_seed_changes_the_search():
    # The seed drives inner folds and model randomness. Final predictions can legitimately coincide
    # (e.g. a seed-free default logistic model wins under both seeds), so compare the search itself.
    X, y = mixed_frame(300, 5, 2)
    a = Tab(max_trials=4, seed=1).fit(X, y).report_.leaderboard
    b = Tab(max_trials=4, seed=2).fit(X, y).report_.leaderboard
    assert [e["mean_log_loss"] for e in a] != [e["mean_log_loss"] for e in b]
    assert np.isfinite([e["mean_log_loss"] for e in a]).all()
