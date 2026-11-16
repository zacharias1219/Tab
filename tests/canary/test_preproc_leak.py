"""US-2 / HF-1: no preprocessing statistic depends on validation or test rows."""

import numpy as np
import pandas as pd
import pytest

from tab.preprocess import Preprocessor, fold_clean
from tests.canary.leaky import LeakyPreprocessor


def _split(binary_frame):
    X, _ = binary_frame
    return X.iloc[:200], X.iloc[200:]


@pytest.mark.parametrize("mode", ["onehot", "ordinal"])
def test_real_preprocessor_is_fold_clean(binary_frame, mode):
    tr, va = _split(binary_frame)
    assert fold_clean(lambda: Preprocessor(mode=mode), tr, va)


@pytest.mark.parametrize("mode", ["onehot", "ordinal"])
def test_canary_leaky_preprocessor_is_caught(binary_frame, mode):
    tr, va = _split(binary_frame)
    # factory receives the validation rows that the check will perturb
    assert not fold_clean(lambda peek=None: LeakyPreprocessor(peek=va, mode=mode), tr, va, pass_peek=True)


def test_statistics_come_only_from_train(binary_frame):
    tr, va = _split(binary_frame)
    p = Preprocessor(mode="onehot").fit(tr)
    assert p.stats_["numeric"]["x2"]["median"] == pytest.approx(float(np.nanmedian(tr["x2"])))
    levels = set(p.stats_["categorical"]["cat"]["levels"])
    assert levels <= set(tr["cat"].dropna().astype(str))


def test_unseen_level_maps_to_other(binary_frame):
    tr, va = _split(binary_frame)
    p = Preprocessor(mode="onehot").fit(tr)
    va2 = va.copy()
    va2["cat"] = pd.Categorical(["zzz"] * len(va2))
    out = p.transform(va2)
    assert np.isfinite(out).all()
