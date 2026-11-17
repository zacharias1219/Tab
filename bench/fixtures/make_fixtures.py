"""Regenerate the two 200-row bundled fixtures (deterministic). Run: python -m bench.fixtures.make_fixtures"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def make(name: str, k: int, seed: int) -> None:
    rng = np.random.RandomState(seed)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-2, 2, size=n)
    x3 = rng.normal(size=n)
    cat = rng.choice(["red", "green", "blue", "grey"], size=n)
    flag = rng.rand(n) < 0.4
    logit = 1.2 * x1 - 0.8 * x2 + 0.6 * (cat == "red") + 0.4 * flag + 0.3 * x3 * x1
    noisy = logit + rng.normal(scale=0.8, size=n)
    if k == 2:
        y = np.where(noisy > 0, "yes", "no")
    else:
        cuts = np.quantile(noisy, [1 / 3, 2 / 3])
        y = np.array(["low", "mid", "high"])[np.digitize(noisy, cuts)]
    x1[rng.rand(n) < 0.08] = np.nan
    cat = cat.astype(object)
    cat[rng.rand(n) < 0.05] = None
    df = pd.DataFrame(
        {
            "x1": x1.round(6),
            "x2": x2.round(6),
            "x3": x3.round(6),
            "colour": cat,
            "flag": flag,
            "__y": y,
        }
    )
    df.to_csv(HERE / f"{name}.csv", index=False)
    dtypes = {
        "x1": "float64",
        "x2": "float64",
        "x3": "float64",
        "colour": "category",
        "flag": "bool",
    }
    (HERE / f"{name}.dtypes.json").write_text(json.dumps(dtypes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    make("fixture_binary", 2, 1)
    make("fixture_multiclass", 3, 2)
