"""Tab with the budget controller off: the fixed-ceiling arm of the controller study (G3)."""

from bench.systems._common import run_system
from bench.systems.tab import make_fit_predict
from tab import __version__

if __name__ == "__main__":
    run_system("tab_fixed", __version__, make_fit_predict(controller_on=False, stacking=True))
