"""Dataset loading, split writing, frozen-list hash (HF-2), dev-pool draw, OpenML split parsing."""

import json

import numpy as np
import pandas as pd
import pytest

from bench import datasets, devpool


@pytest.mark.parametrize("name", ["fixture_binary", "fixture_multiclass"])
def test_fixtures_are_bundled_and_stable(name):
    a = datasets.load(name)
    b = datasets.load(name)
    assert len(a.X) == 200
    pd.testing.assert_frame_equal(a.X, b.X)
    assert len(a.folds) == 5
    for tr, te in a.folds:
        assert not set(tr) & set(te)
    assert sorted(np.concatenate([te for _, te in a.folds]).tolist()) == list(range(200))
    assert a.classes == sorted(a.classes)


def test_write_split_separates_labels(tmp_path):
    ds = datasets.load("fixture_binary")
    sp = datasets.write_split(ds, 0, tmp_path / "s", tmp_path / "l")
    tr = pd.read_parquet(sp.split_dir / "train.parquet")
    te = pd.read_parquet(sp.split_dir / "test_X.parquet")
    assert "__y" in tr.columns and "__y" not in te.columns
    y_test = datasets.read_labels(sp)
    assert len(y_test) == len(te)
    meta = json.loads((sp.label_dir / "classes.json").read_text())
    assert meta["classes"] == ds.classes


def test_frozen_list_hash_roundtrip(tmp_path):
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"status": "frozen", "tier1": [{"task_id": 1, "name": "a"}]}))
    h = datasets.file_sha256(p)
    (tmp_path / "sel.sha256").write_text(h + "\n")
    assert datasets.check_frozen(p, tmp_path / "sel.sha256") is True
    p.write_text(json.dumps({"status": "frozen", "tier1": [{"task_id": 2, "name": "b"}]}))
    with pytest.raises(ValueError, match="HF-2"):
        datasets.check_frozen(p, tmp_path / "sel.sha256")


def test_openml_split_arff_parsing():
    text = """@RELATION splits
@ATTRIBUTE type {TRAIN,TEST}
@ATTRIBUTE rowid NUMERIC
@ATTRIBUTE repeat NUMERIC
@ATTRIBUTE fold NUMERIC
@DATA
TRAIN,0,0,0
TEST,1,0,0
TRAIN,1,0,1
TEST,0,0,1
TRAIN,2,0,0
TRAIN,2,1,0
"""
    folds = datasets.parse_split_arff(text, n_folds=2)
    np.testing.assert_array_equal(folds[0][0], [0, 2])
    np.testing.assert_array_equal(folds[0][1], [1])
    np.testing.assert_array_equal(folds[1][1], [0])


def test_dev_pool_draw_is_deterministic_and_disjoint():
    eligible = list(range(100, 154))
    tiers = list(range(100, 120))
    a = devpool.draw_dev_pool(eligible, tiers, n=12)
    b = devpool.draw_dev_pool(list(reversed(eligible)), tiers, n=12)
    assert a == b
    assert len(a) == 12 and not set(a) & set(tiers)


def test_cache_sha_mismatch_raises(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    man = tmp_path / "manifest.json"
    datasets.record_or_check_sha(man, "k", f)
    f.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        datasets.record_or_check_sha(man, "k", f)
