"""US-4: zoo and leaderboard."""

import pytest

from evalkit.schema import validate
from tab import Tab
from tab.preprocess import Preprocessor
from tab.zoo import FAMILIES, build_model, default_config


def test_zoo_has_the_four_families():
    assert set(FAMILIES) == {"logistic", "random_forest", "hist_gb", "mlp"}


@pytest.mark.parametrize("family", sorted(["logistic", "random_forest", "hist_gb", "mlp"]))
def test_each_model_fits_tiny_data(family, binary_frame):
    X, y = binary_frame
    X, y = X.iloc[:60], y.iloc[:60]
    cfg = default_config(family, n_rows=60, n_features=X.shape[1])
    pre = Preprocessor(mode=FAMILIES[family].preproc_mode).fit(X)
    model = build_model(cfg, seed=0, n_jobs=1)
    model.fit(pre.transform(X), (y == "pos").to_numpy().astype(int))
    p = model.predict_proba(pre.transform(X))
    assert p.shape == (60, 2)


def test_leaderboard_lists_every_trial_with_interval(binary_frame):
    X, y = binary_frame
    m = Tab(max_trials=5, seed=0, stacking=True).fit(X, y)
    lb = m.leaderboard_json()
    validate(lb, "leaderboard")
    entries = lb["entries"]
    assert len(entries) >= 5  # every trial, plus a stack row if one was built
    fams = {e["family"] for e in entries}
    assert {"logistic", "random_forest", "hist_gb", "mlp"} <= fams
    for e in entries:
        assert e["ci_low"] <= e["mean_log_loss"] <= e["ci_high"]
        assert e["wall_s"] >= 0
    ranks = [e["rank"] for e in entries]
    assert ranks == sorted(ranks)
    losses = [e["mean_log_loss"] for e in entries]
    assert losses == sorted(losses)


def test_leaderboard_written_to_disk(binary_frame, tmp_path):
    import json

    X, y = binary_frame
    m = Tab(max_trials=2, seed=0).fit(X, y)
    path = m.write_leaderboard(tmp_path / "leaderboard.json")
    validate(json.loads(path.read_text()), "leaderboard")
