"""Fake system that tries to report its own score in meta.json (HF-3): must be rejected."""
import numpy as np

from bench.systems._common import run_system


def fit_predict(train, y, test, ctx):
    classes = sorted(y.astype(str).unique())
    p = np.full((len(test), len(classes)), 1.0 / len(classes))
    ctx.extra_meta["log_loss"] = 0.0001
    return p, classes, {}


if __name__ == "__main__":
    run_system("self_scorer", "0", fit_predict)
