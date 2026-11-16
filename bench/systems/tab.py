"""Tab under the harness: time mode with the controller on (the default system)."""

import json
import os

from bench.systems._common import run_system
from tab import ControllerConfig, Tab, __version__

TUNED = os.path.join(os.path.dirname(__file__), "..", "..", "prereg", "tuned_params.json")


def tuned() -> dict:
    """Parameters set on the development pool (PREREG section 5); empty means provisional defaults."""
    if os.path.exists(TUNED):
        with open(TUNED, encoding="utf-8") as f:
            return json.load(f)
    return {}


def controller_config(enabled: bool = True) -> ControllerConfig:
    t = tuned()
    if t:
        return ControllerConfig(
            window=int(t["controller_window"]), delta=float(t["controller_delta"]), enabled=enabled
        )
    return ControllerConfig(enabled=enabled)


def make_fit_predict(controller_on: bool = True, stacking: bool = True):
    def fit_predict(train, y, test, ctx):
        t = tuned()
        m = Tab(
            ceiling_s=ctx.remaining_s(),
            seed=ctx.seed,
            n_jobs=ctx.cpus,
            controller=controller_config(controller_on),
            stacking=stacking,
            calibrate=bool(t.get("calibrate", True)),
            leak_threshold=t.get("leak_threshold"),
        )
        m.fit(train, y)
        p = m.predict_proba(test)
        r = m.report_
        log = {
            "stop_reason": r.stop_reason,
            "n_trials": r.n_trials,
            "extra": {
                "chosen": r.chosen,
                "refit_mode": r.refit_mode,
                "calibrated": r.calibrated,
                "ece_before": r.ece_before,
                "ece_after": r.ece_after,
                "fit_wall_s": r.actual_wall_s,
                "n_failed_trials": r.n_failed_trials,
                "leak_flags": r.leak_flags,
                "provisional_params": not bool(t),
                "history": r.history,
            },
        }
        return p, [str(c) for c in m.classes_], log

    return fit_predict


if __name__ == "__main__":
    run_system("tab", __version__, make_fit_predict(True, True))
