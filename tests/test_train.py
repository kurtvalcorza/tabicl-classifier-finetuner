from pathlib import Path
import zipfile

import pandas as pd
import pytest

import train


def _zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


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
