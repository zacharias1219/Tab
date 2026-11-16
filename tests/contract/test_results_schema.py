"""ARCHITECTURE 4.3: results.schema.json rules that carry PRD claims (HF-3, HF-4, HF-5)."""

import pytest

from evalkit.schema import SchemaError, validate
from tests.conftest import make_results, make_run


def ok(doc):
    validate(doc, "results")


def bad(doc):
    with pytest.raises(SchemaError):
        validate(doc, "results")


def test_01_minimal_valid_document(valid_results):
    ok(valid_results)


def test_02_actual_wall_s_required(valid_results):  # HF-5
    del valid_results["runs"][0]["actual_wall_s"]
    bad(valid_results)


def test_03_actual_wall_s_must_be_positive(valid_results):  # HF-5
    valid_results["runs"][0]["actual_wall_s"] = 0
    bad(valid_results)


def test_04_scored_by_must_name_evalkit(valid_results):  # HF-3
    valid_results["runs"][0]["scores"]["scored_by"] = "flaml"
    bad(valid_results)


def test_05_scored_by_required(valid_results):
    del valid_results["runs"][0]["scores"]["scored_by"]
    bad(valid_results)


def test_06_failed_run_needs_error(valid_results):
    valid_results["runs"].append(
        make_run(run_id="tab__credit-g__f1__s0", fold=1, status="error", error=None, scores=None, proba_sha256=None)
    )
    bad(valid_results)


def test_07_failed_run_with_error_is_kept_and_valid(valid_results):
    valid_results["runs"].append(
        make_run(run_id="tab__credit-g__f1__s0", fold=1, status="timeout", error="killed at 1.5x ceiling", scores=None, proba_sha256=None)
    )
    ok(valid_results)


def test_08_failed_run_cannot_carry_scores(valid_results):
    valid_results["runs"][0].update(status="error", error="boom")
    bad(valid_results)


def test_09_ok_run_needs_scores(valid_results):
    valid_results["runs"][0]["scores"] = None
    bad(valid_results)


def test_10_headline_metric_only_skill_score(valid_results):  # HF-4
    valid_results["headline"]["metric"] = "auc"
    bad(valid_results)


def test_11_unknown_top_level_field_rejected(valid_results):
    valid_results["acurracy"] = 0.9
    bad(valid_results)


def test_12_unknown_run_field_rejected(valid_results):
    valid_results["runs"][0]["acutal_wall_s"] = 3.0
    bad(valid_results)


def test_13_unknown_score_field_rejected(valid_results):
    valid_results["runs"][0]["scores"]["accuracy"] = 0.9
    bad(valid_results)


def test_14_status_enum(valid_results):
    valid_results["runs"][0]["status"] = "skipped"
    bad(valid_results)


def test_15_system_name_pattern_blocks_injection(valid_results):
    valid_results["runs"][0]["system"] = "tab; rm -rf /"
    bad(valid_results)


def test_16_dataset_list_hash_required(valid_results):
    del valid_results["dataset_list_sha256"]
    bad(valid_results)


def test_17_budget_ratio_required(valid_results):  # NFR-1 flagged per run
    del valid_results["runs"][0]["budget_ratio"]
    bad(valid_results)


def test_18_nominal_budget_alone_is_not_enough():  # HF-5: budget without actual seconds
    run = make_run()
    del run["actual_wall_s"]
    bad(make_results(runs=[run]))
