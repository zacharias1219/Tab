"""FLAML baseline (bench extra only). Budget: the same ceiling minus time already spent."""

import numpy as np

from bench.systems._common import run_system


def fit_predict(train, y, test, ctx):
    from flaml import AutoML

    automl = AutoML()
    automl.fit(
        X_train=train,
        y_train=y,
        task="classification",
        metric="log_loss",  # FLAML's own search objective; scoring is still evalkit's alone
        time_budget=ctx.remaining_s(margin_s=1.0, margin_frac=0.05),
        n_jobs=ctx.cpus,
        seed=ctx.seed,
        verbose=0,
        retrain_full=True,
    )
    p = automl.predict_proba(test)
    classes = [str(c) for c in automl.classes_]
    order = np.argsort(classes)
    return (
        np.asarray(p, dtype=np.float64)[:, order],
        [classes[i] for i in order],
        {
            "stop_reason": None,
            "n_trials": None,
            "extra": {"best_estimator": automl.best_estimator},
        },
    )


if __name__ == "__main__":
    import flaml

    run_system("flaml", flaml.__version__, fit_predict)
