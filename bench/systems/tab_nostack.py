"""Tab without stacking: the ablation arm (PREREG section 9)."""

from bench.systems._common import run_system
from bench.systems.tab import make_fit_predict
from tab import __version__

if __name__ == "__main__":
    run_system("tab_nostack", __version__, make_fit_predict(controller_on=True, stacking=False))
