"""US-8: the 30 pre-specified leakage cases (4 types x 3 severities x 2 tables + 6 negatives).

The kill criterion (pooled simple-set recall >= 70%) is asserted here on the two CI tables; the
240-case study on the frozen datasets lives in bench/leak_study.py.
"""

import numpy as np
import pytest
from sklearn.datasets import load_wine

from bench.leak_inject import SEVERITIES, SIMPLE_SET, TYPES, inject, inject_null_column, is_detected
from tab.leak import detect_leaks
from tests.conftest import mixed_frame


def _tables():
    Xs, ys = mixed_frame(600, 7, 2)
    Xw, yw = load_wine(return_X_y=True, as_frame=True)
    yw = yw.map(lambda v: f"cultivar{v}")
    out = {}
    for name, (X, y) in {"synthetic": (Xs, ys), "wine": (Xw, yw)}.items():
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(X))
        cut = int(0.7 * len(X))
        tr, te = perm[:cut], perm[cut:]
        out[name] = (
            X.iloc[tr].reset_index(drop=True),
            y.iloc[tr].reset_index(drop=True),
            X.iloc[te].reset_index(drop=True),
            y.iloc[te].reset_index(drop=True),
        )
    return out


TABLES = _tables()
CASES = [(t, ty, sev) for t in TABLES for ty in TYPES for sev in SEVERITIES[ty]]


def test_case_count_is_30():
    assert len(CASES) == 24
    assert len(CASES) + 6 == 30


def _run(table, ty, sev):
    Xtr, ytr, Xte, yte = TABLES[table]
    Xtr2, ytr2, Xte2, expected = inject(Xtr, ytr, Xte, yte, ty, sev, seed=20261116)
    rep = detect_leaks(Xtr2, ytr2, X_test=Xte2)
    return is_detected(rep, expected)


@pytest.fixture(scope="module")
def outcomes():
    return {(t, ty, sev): _run(t, ty, sev) for t, ty, sev in CASES}


def test_kill_criterion_simple_set_recall(outcomes):
    simple = [v for (t, ty, sev), v in outcomes.items() if (ty, sev) in SIMPLE_SET]
    assert len(simple) == 8
    assert np.mean(simple) >= 0.70


@pytest.mark.parametrize("table", sorted(TABLES))
def test_target_copies_all_detected(outcomes, table):
    assert all(outcomes[(table, "T1", s)] for s in SEVERITIES["T1"])


@pytest.mark.parametrize("table", sorted(TABLES))
def test_mild_near_copy_detected(outcomes, table):
    assert outcomes[(table, "T2", SEVERITIES["T2"][0])]


@pytest.mark.parametrize("table", sorted(TABLES))
def test_large_overlap_detected(outcomes, table):
    assert outcomes[(table, "T4", "20%")]
    assert outcomes[(table, "T4", "5%")]


@pytest.mark.parametrize("table", sorted(TABLES))
def test_fully_sorted_detected(outcomes, table):
    assert outcomes[(table, "T3", "fully_sorted")]


@pytest.mark.parametrize("table", sorted(TABLES))
@pytest.mark.parametrize("kind", ["normal", "uniform", "categorical"])
def test_clean_negatives_not_flagged(table, kind):
    Xtr, ytr, Xte, _ = TABLES[table]
    X2, col = inject_null_column(Xtr, kind, seed=3)
    rep = detect_leaks(X2, ytr, X_test=None)
    assert col not in rep.flagged_columns()


def test_report_states_null_false_positive_rate():
    Xtr, ytr, Xte, _ = TABLES["synthetic"]
    rep = detect_leaks(Xtr, ytr, X_test=Xte, null_columns=5, seed=0)
    assert rep.null_fpr is not None and 0.0 <= rep.null_fpr <= 1.0
    assert rep.null_n == 5
    for f in rep.flags:
        assert set(f) >= {"column", "type", "score", "flagged"}
