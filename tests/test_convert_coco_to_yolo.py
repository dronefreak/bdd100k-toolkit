"""Tests for the canonical-COCO -> Ultralytics YOLO bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from bdd100k_toolkit.detection import get
from bdd100k_toolkit.utils.convert_coco_to_yolo import convert


def _args(input_dir: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=str(input_dir), output_dir=str(output_dir), copy_images=False
    )


@pytest.fixture
def coco_dir(raw_bdd100k_detection_dir: Path) -> Path:
    out = raw_bdd100k_detection_dir.parent / "coco_out"
    with pytest.warns(UserWarning):
        get("bdd100k-detection").prepare_coco(raw_bdd100k_detection_dir, out)
    return out


def test_convert_writes_labels_and_relocatable_yaml(
    coco_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "yolo"
    convert(_args(coco_dir, out))

    yaml_text = (out / "data.yaml").read_text()
    assert "path:" not in yaml_text  # resolved relative to the yaml itself
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
    assert "test: images/test" in yaml_text
    assert "nc: 10" in yaml_text
    assert (out / "labels" / "test" / "v0.txt").read_text().startswith("4 ")  # bus


def test_convert_omits_missing_splits_from_yaml(coco_dir: Path, tmp_path: Path) -> None:
    (coco_dir / "test" / "_annotations.coco.json").unlink()
    out = tmp_path / "yolo"
    convert(_args(coco_dir, out))
    yaml_text = (out / "data.yaml").read_text()
    assert "test:" not in yaml_text
    assert "train:" in yaml_text


def test_convert_raises_when_nothing_to_convert(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert(_args(tmp_path / "empty", tmp_path / "yolo"))


def test_convert_rejects_inconsistent_categories(
    coco_dir: Path, tmp_path: Path
) -> None:
    path = coco_dir / "test" / "_annotations.coco.json"
    data = json.loads(path.read_text())
    data["categories"][0]["name"] = "renamed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different categories"):
        convert(_args(coco_dir, tmp_path / "yolo"))


def test_convert_warns_about_stale_images(coco_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "yolo"
    convert(_args(coco_dir, out))
    (out / "images" / "test" / "stale.jpg").write_bytes(b"x")
    with pytest.warns(UserWarning, match="not in"):
        convert(_args(coco_dir, out))
