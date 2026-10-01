"""Adapters must fail loudly instead of writing empty or partial splits."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bdd100k_toolkit.classification.datasets.weather import BDD100KWeatherAdapter
from bdd100k_toolkit.detection import get as get_detection
from bdd100k_toolkit.segmentation.semantic import get as get_segmentation
from bdd100k_toolkit.utils.checks import EmptySplitError, require_nonempty


def test_require_nonempty_raises_when_required() -> None:
    with pytest.raises(EmptySplitError, match="train"):
        require_nonempty("train", 0, "nothing matched")


def test_require_nonempty_warns_when_optional() -> None:
    with pytest.warns(UserWarning, match="valid"):
        require_nonempty("valid", 0, "nothing matched", required=False)


def test_require_nonempty_passes_when_nonempty() -> None:
    require_nonempty("train", 3, "unused")


def test_classification_raises_when_no_images_match(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    shutil.rmtree(raw_bdd100k_dir / "images" / "100k" / "train")
    (raw_bdd100k_dir / "images" / "100k" / "train").mkdir()
    with pytest.raises(EmptySplitError, match="train"):
        BDD100KWeatherAdapter().prepare_classification(
            raw_bdd100k_dir, tmp_path / "out"
        )


def test_detection_raises_when_no_images_match(
    raw_bdd100k_detection_dir: Path, tmp_path: Path
) -> None:
    shutil.rmtree(raw_bdd100k_detection_dir / "images" / "100k" / "val")
    (raw_bdd100k_detection_dir / "images" / "100k" / "val").mkdir()
    with pytest.raises(EmptySplitError, match="test"):
        get_detection("bdd100k-detection").prepare_coco(
            raw_bdd100k_detection_dir, tmp_path / "out"
        )


def test_detection_raises_without_label_files(
    raw_bdd100k_detection_dir: Path, tmp_path: Path
) -> None:
    for path in (raw_bdd100k_detection_dir / "labels").glob("*.json"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="det_20"):
        get_detection("bdd100k-detection").prepare_coco(
            raw_bdd100k_detection_dir, tmp_path / "out"
        )


def test_segmentation_raises_when_masks_do_not_match(
    raw_bdd100k_semantic_seg_dir: Path, tmp_path: Path
) -> None:
    mask_dir = raw_bdd100k_semantic_seg_dir / "labels" / "sem_seg" / "masks" / "train"
    for mask in mask_dir.glob("*.png"):
        mask.rename(mask.with_name(f"other_{mask.name}"))
    with pytest.raises(EmptySplitError, match="train"):
        get_segmentation("bdd100k-semantic-seg").prepare_segmentation(
            raw_bdd100k_semantic_seg_dir, tmp_path / "out"
        )


def test_segmentation_accepts_legacy_train_id_mask_names(
    raw_bdd100k_semantic_seg_dir: Path, tmp_path: Path
) -> None:
    for split in ("train", "val"):
        mask_dir = raw_bdd100k_semantic_seg_dir / "labels" / "sem_seg" / "masks" / split
        for mask in mask_dir.glob("*.png"):
            mask.rename(mask.with_name(f"{mask.stem}_train_id.png"))
    out = tmp_path / "out"
    get_segmentation("bdd100k-semantic-seg").prepare_segmentation(
        raw_bdd100k_semantic_seg_dir, out
    )
    assert len(list((out / "test" / "masks").glob("*.png"))) == 2
    # Canonical output always uses the plain stem.
    assert (out / "test" / "masks" / "v0.png").is_file()


def _write_det20(raw_dir: Path, names: dict[str, str]) -> None:
    """Replace legacy labels with det_20-style labels (pedestrian/bicycle/...)."""
    labels = raw_dir / "labels"
    for path in labels.glob("*.json"):
        path.unlink()
    (labels / "det_20").mkdir()
    entries = {
        "train": [
            {
                "name": f"t{i}.jpg",
                "labels": [
                    {
                        "category": names["person"],
                        "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 20},
                        "attributes": {"occluded": True, "crowd": i == 0},
                    },
                    {
                        "category": names["bike"],
                        "box2d": {"x1": 5, "y1": 5, "x2": 9, "y2": 9},
                    },
                    {
                        "category": "other vehicle",
                        "box2d": {"x1": 5, "y1": 5, "x2": 9, "y2": 9},
                    },
                    {
                        "category": "van",
                        "box2d": {"x1": 5, "y1": 5, "x2": 9, "y2": 9},
                    },
                ],
            }
            for i in range(6)
        ],
        "val": [
            {
                "name": f"v{i}.jpg",
                "labels": [
                    {
                        "category": names["motor"],
                        "box2d": {"x1": 0, "y1": 0, "x2": 8, "y2": 8},
                    }
                ],
            }
            for i in range(2)
        ],
    }
    for split, split_entries in entries.items():
        (labels / "det_20" / f"det_{split}.json").write_text(json.dumps(split_entries))


@pytest.mark.parametrize(
    "names",
    [
        {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"},
        {"person": "person", "bike": "bike", "motor": "motor"},
    ],
)
def test_detection_reads_det20_with_either_naming_scheme(
    raw_bdd100k_detection_dir: Path, tmp_path: Path, names: dict[str, str]
) -> None:
    _write_det20(raw_bdd100k_detection_dir, names)
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="unknown categories"):
        get_detection("bdd100k-detection").prepare_coco(raw_bdd100k_detection_dir, out)

    train = json.loads((out / "train" / "_annotations.coco.json").read_text())
    category_names = {c["id"]: c["name"] for c in train["categories"]}
    used = {category_names[a["category_id"]] for a in train["annotations"]}
    assert used == {"person", "bike"}  # "other vehicle" ignored, "van" unknown

    test = json.loads((out / "test" / "_annotations.coco.json").read_text())
    assert {category_names[a["category_id"]] for a in test["annotations"]} == {"motor"}


def test_detection_keeps_crowd_and_attributes(
    raw_bdd100k_detection_dir: Path, tmp_path: Path
) -> None:
    _write_det20(
        raw_bdd100k_detection_dir,
        {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"},
    )
    out = tmp_path / "out"
    with pytest.warns(UserWarning):
        get_detection("bdd100k-detection").prepare_coco(raw_bdd100k_detection_dir, out)
    annotations = []
    for split in ("train", "valid"):
        data = json.loads((out / split / "_annotations.coco.json").read_text())
        annotations += data["annotations"]
    persons = [a for a in annotations if a["category_id"] == 0]
    assert any(a["iscrowd"] == 1 for a in persons)
    assert all(a["attributes"] == {"occluded": True} for a in persons)


def test_detection_prefers_det20_over_legacy(
    raw_bdd100k_detection_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    labels = raw_bdd100k_detection_dir / "labels"
    (labels / "det_20").mkdir()
    for split in ("train", "val"):
        shutil.copy(
            labels / f"bdd100k_labels_images_{split}.json",
            labels / "det_20" / f"det_{split}.json",
        )
    with pytest.warns(UserWarning):
        get_detection("bdd100k-detection").prepare_coco(
            raw_bdd100k_detection_dir, tmp_path / "out"
        )
    assert "Using det_20" in capsys.readouterr().out


def test_detection_reads_image_size_from_file(
    raw_bdd100k_detection_dir: Path, tmp_path: Path
) -> None:
    Image.new("RGB", (64, 32)).save(
        raw_bdd100k_detection_dir / "images" / "100k" / "val" / "v0.jpg"
    )
    out = tmp_path / "out"
    with pytest.warns(UserWarning):  # fake-bytes images fall back, and are reported
        get_detection("bdd100k-detection").prepare_coco(raw_bdd100k_detection_dir, out)
    test = json.loads((out / "test" / "_annotations.coco.json").read_text())
    sizes = {i["file_name"]: (i["width"], i["height"]) for i in test["images"]}
    assert sizes["v0.jpg"] == (64, 32)
    assert sizes["v1.jpg"] == (1280, 720)


def test_detection_image_ids_are_contiguous(
    raw_bdd100k_detection_dir: Path, tmp_path: Path
) -> None:
    (raw_bdd100k_detection_dir / "images" / "100k" / "val" / "v0.jpg").unlink()
    out = tmp_path / "out"
    with pytest.warns(UserWarning):
        get_detection("bdd100k-detection").prepare_coco(raw_bdd100k_detection_dir, out)
    test = json.loads((out / "test" / "_annotations.coco.json").read_text())
    assert [i["id"] for i in test["images"]] == [0]
    assert np.all([a["image_id"] == 0 for a in test["annotations"]])
