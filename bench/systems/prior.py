"""Class-prior stub: predicts training-fold class frequencies. Skill 0 by construction."""

import numpy as np

from bench.systems._common import run_system


def fit_predict(train, y, test, ctx):
    classes = sorted(y.unique())
    freq = y.value_counts(normalize=True).reindex(classes).to_numpy(dtype=np.float64)
    return np.tile(freq, (len(test), 1)), classes, {"stop_reason": None, "n_trials": 0}


if __name__ == "__main__":
    run_system("prior", "1", fit_predict)
