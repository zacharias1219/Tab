"""US-14: the results page build fails if any publish-standard section lacks data."""

import pytest

from report.build import REQUIRED_SECTIONS, MissingSectionData, build_page


def test_builds_with_complete_data(tmp_path, valid_results):
    valid_results["analyses"]["cost"] = [{"system": "tab", "median_actual_s": 58.2, "max_peak_rss_gb": 0.4}]
    valid_results["analyses"]["per_dataset"] = [{"system": "tab", "dataset": "credit-g", "mean_skill": 0.2}]
    out = build_page(valid_results, tmp_path)
    html = (out / "index.html").read_text(encoding="utf-8")
    for sec in REQUIRED_SECTIONS:
        assert f'id="{sec}"' in html
    assert "not yet a claim" in html
    assert (out / "results.json").exists()


@pytest.mark.parametrize("hole", ["runs", "cost", "per_dataset"])
def test_missing_section_data_fails(tmp_path, valid_results, hole):
    valid_results["analyses"]["cost"] = [{"system": "tab", "median_actual_s": 1.0, "max_peak_rss_gb": 0.1}]
    valid_results["analyses"]["per_dataset"] = [{"system": "tab", "dataset": "d", "mean_skill": 0.1}]
    if hole == "runs":
        valid_results["runs"] = []
    else:
        valid_results["analyses"][hole] = []
    with pytest.raises(MissingSectionData):
        build_page(valid_results, tmp_path)


def test_full_mode_requires_studies(tmp_path, valid_results):
    valid_results["config"]["mode"] = "full"
    valid_results["analyses"]["cost"] = [{"system": "tab", "median_actual_s": 1.0, "max_peak_rss_gb": 0.1}]
    valid_results["analyses"]["per_dataset"] = [{"system": "tab", "dataset": "d", "mean_skill": 0.1}]
    with pytest.raises(MissingSectionData, match="leak_study"):
        build_page(valid_results, tmp_path)


def test_invalid_results_rejected(tmp_path, valid_results):
    del valid_results["runs"][0]["actual_wall_s"]
    with pytest.raises(Exception):
        build_page(valid_results, tmp_path)


def test_page_prints_tie_from_headline(tmp_path, valid_results):  # HF-4 at the page level
    valid_results["analyses"]["cost"] = [{"system": "tab", "median_actual_s": 1.0, "max_peak_rss_gb": 0.1}]
    valid_results["analyses"]["per_dataset"] = [{"system": "tab", "dataset": "d", "mean_skill": 0.1}]
    valid_results["headline"]["statement"] = "tab skill score (log loss) vs flaml: tie (interval includes zero)."
    out = build_page(valid_results, tmp_path)
    assert "tie (interval includes zero)" in (out / "index.html").read_text(encoding="utf-8")
