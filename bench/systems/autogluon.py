"""AutoGluon baseline (bench extra only). presets='medium_quality' per PREREG section 3."""

import shutil
import tempfile

import numpy as np

from bench.systems._common import run_system


def fit_predict(train, y, test, ctx):
    from autogluon.tabular import TabularPredictor

    tmp = tempfile.mkdtemp(prefix="ag_")
    try:
        data = train.copy()
        data["__y"] = y.to_numpy()
        pred = TabularPredictor(label="__y", eval_metric="log_loss", path=tmp, verbosity=0)
        pred.fit(
            data,
            time_limit=ctx.remaining_s(margin_s=2.0, margin_frac=0.08),
            presets="medium_quality",
            num_cpus=ctx.cpus,
            num_gpus=0,
        )
        p = pred.predict_proba(test)
        classes = sorted(str(c) for c in p.columns)
        p.columns = [str(c) for c in p.columns]
        out = p[classes].to_numpy(dtype=np.float64)
        out = out / out.sum(axis=1, keepdims=True)
        return (
            out,
            classes,
            {"stop_reason": None, "n_trials": None, "extra": {"best_model": pred.model_best}},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import autogluon.tabular as agt

    run_system("autogluon", agt.__version__, fit_predict)
