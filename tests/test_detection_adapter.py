"""Tests for the BDD100K detection adapter's ``prepare_coco``."""

from __future__ import annotations

import json
from pathlib import Path

from bdd100k_toolkit.detection import get


def test_prepare_coco_writes_expected_splits(raw_bdd100k_detection_dir: Path) -> None:
    output_dir = raw_bdd100k_detection_dir.parent / "coco_out"
    adapter = get("bdd100k-detection")
    adapter.prepare_coco(raw_bdd100k_detection_dir, output_dir)

    for split in ("train", "valid", "test"):
        annotation_path = output_dir / split / "_annotations.coco.json"
        assert annotation_path.is_file()

    # 6 train entries total; a seeded 15% split carves >=1 into valid.
    train_data = json.loads(
        (output_dir / "train" / "_annotations.coco.json").read_text()
    )
    valid_data = json.loads(
        (output_dir / "valid" / "_annotations.coco.json").read_text()
    )
    assert len(train_data["images"]) + len(valid_data["images"]) == 6

    test_data = json.loads((output_dir / "test" / "_annotations.coco.json").read_text())
    assert len(test_data["images"]) == 2

    # 10 classes always emitted regardless of which ones appear in this split.
    assert len(train_data["categories"]) == 10


def test_prepare_coco_drops_unknown_categories_and_bad_categories(
    raw_bdd100k_detection_dir: Path,
) -> None:
    output_dir = raw_bdd100k_detection_dir.parent / "coco_out"
    adapter = get("bdd100k-detection")
    adapter.prepare_coco(raw_bdd100k_detection_dir, output_dir)

    all_annotations = []
    for split in ("train", "valid", "test"):
        data = json.loads((output_dir / split / "_annotations.coco.json").read_text())
        all_annotations.extend(data["annotations"])

    category_ids = {ann["category_id"] for ann in all_annotations}
    # "bad-category" in t5.jpg must never resolve to a category id.
    assert all(0 <= cid < 10 for cid in category_ids)
    # 2 (t0) + 1 (t1) + 1 (t2) + 0 (t3) + 1 (t4) + 1 (t5, "bad-category" dropped)
    # + 1 (v0) + 1 (v1) = 8 boxes total.
    assert len(all_annotations) == 8


def test_prepare_coco_links_images(raw_bdd100k_detection_dir: Path) -> None:
    output_dir = raw_bdd100k_detection_dir.parent / "coco_out"
    adapter = get("bdd100k-detection")
    adapter.prepare_coco(raw_bdd100k_detection_dir, output_dir)

    assert (output_dir / "test" / "v0.jpg").is_file()
    assert (output_dir / "test" / "v1.jpg").is_file()
