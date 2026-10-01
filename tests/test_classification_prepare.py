"""Classification prepare: input layouts, the `unknown` class, shared splits."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bdd100k_toolkit.classification import get
from bdd100k_toolkit.detection import get as get_detection
from bdd100k_toolkit.utils.checks import EmptySplitError

_CLASS_DIR = {"undefined": "unknown", "dawn/dusk": "dawn or dusk"}


def _listing(root: Path) -> set[tuple[str, str, str]]:
    """``{(split, class, file name)}`` for every image under a canonical root."""
    return {(p.parts[-3], p.parts[-2], p.name) for p in root.glob("*/*/*.jpg")}


def _as_kaggle_folders(raw_dir: Path, attribute: str, out: Path) -> Path:
    """Re-express the fake official download as Kaggle ``{train,val}/<class>/``."""
    for split in ("train", "val"):
        entries = json.loads(
            (raw_dir / "labels" / f"bdd100k_labels_images_{split}.json").read_text()
        )
        for entry in entries:
            raw = entry["attributes"][attribute]
            class_dir = out / split / _CLASS_DIR.get(raw, raw)
            class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(raw_dir / "images" / "100k" / split / entry["name"], class_dir)
    (out / "test").mkdir()
    (out / "test" / "unlabeled.jpg").write_bytes(b"fake")
    return out


@pytest.mark.parametrize(
    ("key", "attribute"),
    [
        ("bdd100k-weather", "weather"),
        ("bdd100k-period", "timeofday"),
        ("bdd100k-scenario", "scene"),
    ],
)
def test_both_layouts_give_identical_output(
    raw_bdd100k_dir: Path, tmp_path: Path, key: str, attribute: str
) -> None:
    folders = _as_kaggle_folders(raw_bdd100k_dir, attribute, tmp_path / "kaggle")
    from_official = tmp_path / "from_official"
    from_folders = tmp_path / "from_folders"
    get(key).prepare_classification(raw_bdd100k_dir, from_official)
    get(key).prepare_classification(folders, from_folders)
    assert _listing(from_official) == _listing(from_folders)
    assert not any("unlabeled" in str(p) for p in from_folders.rglob("*.jpg"))


def test_exclude_unknown_keeps_the_same_holdout(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    with_unknown = tmp_path / "with"
    without = tmp_path / "without"
    adapter = get("bdd100k-weather")
    adapter.prepare_classification(raw_bdd100k_dir, with_unknown)
    adapter.prepare_classification(raw_bdd100k_dir, without, include_unknown=False)
    valid_with = {n for s, _, n in _listing(with_unknown) if s == "valid"}
    valid_without = {n for s, _, n in _listing(without) if s == "valid"}
    assert valid_without == valid_with - {"t4.jpg"}


def test_dawn_or_dusk_is_one_folder_not_a_nested_path(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    get("bdd100k-period").prepare_classification(raw_bdd100k_dir, out)
    assert not list(out.rglob("dawn"))  # would exist if "dawn/dusk" became a path
    assert list(out.glob("*/dawn or dusk/t2.jpg"))


def test_labels_dir_can_live_elsewhere(raw_bdd100k_dir: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "separately_extracted_labels"
    shutil.move(str(raw_bdd100k_dir / "labels"), elsewhere)
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        get("bdd100k-weather").prepare_classification(raw_bdd100k_dir, out)
    get("bdd100k-weather").prepare_classification(
        raw_bdd100k_dir, out, labels_dir=elsewhere
    )
    assert list(out.glob("test/foggy/v0.jpg"))


def test_unrecognised_layout_is_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "something").mkdir()
    with pytest.raises(FileNotFoundError, match="Unrecognised layout"):
        get("bdd100k-weather").prepare_classification(tmp_path, tmp_path / "out")


def test_folders_of_another_task_are_rejected(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    folders = _as_kaggle_folders(raw_bdd100k_dir, "weather", tmp_path / "kaggle")
    with pytest.raises(ValueError, match="right task"):
        get("bdd100k-scenario").prepare_classification(folders, tmp_path / "out")


def test_folders_layout_can_exclude_unknown(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    folders = _as_kaggle_folders(raw_bdd100k_dir, "weather", tmp_path / "kaggle")
    out = tmp_path / "out"
    get("bdd100k-weather").prepare_classification(folders, out, include_unknown=False)
    assert not list(out.glob("*/unknown"))
    assert len(_listing(out)) == 7  # 5 train/valid + 2 test


def test_unexpected_attribute_values_are_reported(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    path = raw_bdd100k_dir / "labels" / "bdd100k_labels_images_train.json"
    entries = json.loads(path.read_text())
    entries[0]["attributes"]["weather"] = "sunny"
    path.write_text(json.dumps(entries))
    with pytest.warns(UserWarning, match="unexpected value 'sunny'"):
        get("bdd100k-weather").prepare_classification(raw_bdd100k_dir, tmp_path / "o")


def test_labels_without_image_files_are_reported(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    (raw_bdd100k_dir / "images" / "100k" / "train" / "t0.jpg").unlink()
    with pytest.warns(UserWarning, match="label without image file"):
        get("bdd100k-weather").prepare_classification(raw_bdd100k_dir, tmp_path / "o")


def test_empty_test_split_raises(raw_bdd100k_dir: Path, tmp_path: Path) -> None:
    shutil.rmtree(raw_bdd100k_dir / "images" / "100k" / "val")
    (raw_bdd100k_dir / "images" / "100k" / "val").mkdir()
    with pytest.raises(EmptySplitError, match="test"), pytest.warns(UserWarning):
        get("bdd100k-weather").prepare_classification(raw_bdd100k_dir, tmp_path / "o")


def test_classification_and_detection_share_the_same_holdout(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    """Same images are valid/test in every task, so results are comparable."""
    for split in ("train", "val"):
        path = raw_bdd100k_dir / "labels" / f"bdd100k_labels_images_{split}.json"
        entries = json.loads(path.read_text())
        for entry in entries:
            entry["labels"] = [
                {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 4, "y2": 4}}
            ]
        path.write_text(json.dumps(entries))
    cls_out = tmp_path / "cls"
    det_out = tmp_path / "det"
    get("bdd100k-weather").prepare_classification(raw_bdd100k_dir, cls_out)
    with pytest.warns(UserWarning):
        get_detection("bdd100k-detection").prepare_coco(raw_bdd100k_dir, det_out)
    for split in ("train", "valid", "test"):
        cls_names = {n for s, _, n in _listing(cls_out) if s == split}
        det_names = {p.name for p in (det_out / split).glob("*.jpg")}
        assert cls_names == det_names, split
