"""Fake system whose probabilities do not sum to 1."""
import numpy as np

from bench.systems._common import run_system


def fit_predict(train, y, test, ctx):
    classes = sorted(y.astype(str).unique())
    return np.full((len(test), len(classes)), 0.9), classes, {}


if __name__ == "__main__":
    run_system("bad_rows", "0", fit_predict)
