import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

import train


@pytest.fixture(autouse=True)
def _reset_limits():
    train._load_limits()
    yield


def _zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_load_limits_rejects_malformed_env(monkeypatch):
    monkeypatch.setenv("DIMER_MAX_SINGLE_CSV_BYTES", "not-an-int")
    with pytest.raises(ValueError):
        train._load_limits()


def test_main_writes_failure_on_malformed_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMER_MAX_SINGLE_CSV_BYTES", "not-an-int")
    monkeypatch.setattr(train, "RESULT_PATH", tmp_path / "result.json")
    assert train.main() == 1
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["successful"] is False


def test_batched_predict_chunks_and_matches(monkeypatch):
    monkeypatch.setattr(train, "PREDICT_BATCH_ROWS", 3)
    X = pd.DataFrame({"a": range(10)})
    calls = []

    def fake(sub):
        calls.append(len(sub))
        return sub["a"].to_numpy()

    out = train._batched(fake, X)
    assert list(out) == list(range(10))
    assert calls == [3, 3, 3, 1]


def test_stratified_holdout_preserves_classes():
    frame = pd.DataFrame({"x": range(100), "target": ["a"] * 70 + ["b"] * 30})
    tr, va = train._stratified_holdout(frame, "target", 0.2, 0)
    assert set(tr.target) == {"a", "b"}
    assert set(va.target) == {"a", "b"}
    assert len(tr) + len(va) == 100


def test_stratified_holdout_rejects_singleton_class():
    frame = pd.DataFrame({"x": range(10), "target": ["a"] * 9 + ["b"]})
    with pytest.raises(ValueError, match="at least 2"):
        train._stratified_holdout(frame, "target", 0.2, 0)


def test_categorical_encoder_ordinal_numeric_passthrough_and_unknown():
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["b", "a", "b"],
            "flag": [True, False, True],
            "target": [0, 1, 0],
        }
    )
    enc = train._fit_categorical_encoder(frame, ["num", "cat", "flag"])
    assert "num" not in enc  # numeric passthrough
    assert enc["cat"] == ["a", "b"]  # sorted categories
    assert enc["flag"] == ["False", "True"]

    out = train._apply_categorical_encoder(frame, enc)
    assert list(out["cat"]) == [1, 0, 1]  # b=1, a=0
    assert list(out["flag"]) == [1, 0, 1]  # True=1, False=0
    assert list(out["num"]) == [1.0, 2.0, 3.0]  # unchanged

    # unseen category and missing value both fall into the unknown bucket (len)
    infer = pd.DataFrame({"num": [9.0, 9.0], "cat": ["z", None], "flag": [True, None]})
    enc_infer = train._apply_categorical_encoder(infer, enc)
    assert list(enc_infer["cat"]) == [2, 2]  # z unseen -> 2, None -> 2
    assert list(enc_infer["flag"]) == [1, 2]  # True -> 1, None -> 2 (unknown)


def test_categorical_encoder_empty_when_all_numeric():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4], "target": [0, 1]})
    enc = train._fit_categorical_encoder(frame, ["a", "b"])
    assert enc == {}
    assert train._apply_categorical_encoder(frame, enc) is frame  # no-op passthrough


def test_duplicate_train_is_rejected(tmp_path, monkeypatch):
    _zip(
        tmp_path / "dataset.zip",
        {
            "a/train.csv": "x,target\n1,a\n2,b\n",
            "b/train.csv": "x,target\n3,a\n4,b\n",
        },
    )
    monkeypatch.setattr(train, "DATASET_DIR", tmp_path)
    with pytest.raises(ValueError, match="multiple train.csv"):
        train._read_csv("train")


def test_normalize_member_rejects_traversal_and_absolute():
    assert train._normalize_member("train.csv") == "train.csv"
    assert train._normalize_member("./train.csv") == "train.csv"
    assert train._normalize_member("dataset/train.csv") == "train.csv"
    for hostile in ("../train.csv", "../../etc/passwd", "/train.csv", "dataset/../secret.csv"):
        with pytest.raises(ValueError, match="unsafe archive member"):
            train._normalize_member(hostile)


def test_prepare_frames_rejects_target_in_drop_columns():
    with pytest.raises(ValueError, match="must not appear in drop_columns"):
        train._prepare_frames({"target_column": "target", "drop_columns": "target,x"}, 0)


def test_stratified_cap_rejects_cap_below_class_count():
    frame = pd.DataFrame({"x": range(10), "target": [f"c{i}" for i in range(10)]})
    with pytest.raises(ValueError, match="below the class count"):
        train._stratified_cap(frame, "target", 5, 0)
