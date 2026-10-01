"""Tests for the BDD100K semantic segmentation adapter's ``prepare_segmentation``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bdd100k_toolkit.segmentation.semantic import get


def test_prepare_segmentation_writes_expected_splits(
    raw_bdd100k_semantic_seg_dir: Path,
) -> None:
    output_dir = raw_bdd100k_semantic_seg_dir.parent / "seg_out"
    adapter = get("bdd100k-semantic-seg")
    adapter.prepare_segmentation(raw_bdd100k_semantic_seg_dir, output_dir)

    for split in ("train", "valid", "test"):
        assert (output_dir / split / "images").is_dir()
        assert (output_dir / split / "masks").is_dir()

    train_images = list((output_dir / "train" / "images").glob("*.jpg"))
    valid_images = list((output_dir / "valid" / "images").glob("*.jpg"))
    test_images = list((output_dir / "test" / "images").glob("*.jpg"))

    # 6 train entries total; a seeded 15% split carves >=1 into valid.
    assert len(train_images) + len(valid_images) == 6
    assert len(test_images) == 2


def test_prepare_segmentation_pairs_masks_with_images(
    raw_bdd100k_semantic_seg_dir: Path,
) -> None:
    output_dir = raw_bdd100k_semantic_seg_dir.parent / "seg_out"
    adapter = get("bdd100k-semantic-seg")
    adapter.prepare_segmentation(raw_bdd100k_semantic_seg_dir, output_dir)

    for image_path in (output_dir / "test" / "images").glob("*.jpg"):
        mask_path = output_dir / "test" / "masks" / f"{image_path.stem}.png"
        assert mask_path.is_file()


def test_prepare_segmentation_preserves_mask_pixel_values(
    raw_bdd100k_semantic_seg_dir: Path,
) -> None:
    output_dir = raw_bdd100k_semantic_seg_dir.parent / "seg_out"
    adapter = get("bdd100k-semantic-seg")
    adapter.prepare_segmentation(raw_bdd100k_semantic_seg_dir, output_dir)

    mask_path = output_dir / "test" / "masks" / "v0.png"
    mask = np.asarray(Image.open(mask_path))
    assert mask.shape == (8, 8)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0}


def test_prepare_segmentation_keeps_ignore_pixel_value(
    raw_bdd100k_semantic_seg_dir: Path,
) -> None:
    output_dir = raw_bdd100k_semantic_seg_dir.parent / "seg_out"
    adapter = get("bdd100k-semantic-seg")
    adapter.prepare_segmentation(raw_bdd100k_semantic_seg_dir, output_dir)

    # Every train-source mask has pixel (0, 0) set to the ignore value.
    found_ignore = False
    for split in ("train", "valid"):
        for mask_path in (output_dir / split / "masks").glob("*.png"):
            mask = np.asarray(Image.open(mask_path))
            if mask[0, 0] == 255:
                found_ignore = True
    assert found_ignore


def test_prepare_segmentation_raises_on_missing_raw_dirs(tmp_path: Path) -> None:
    adapter = get("bdd100k-semantic-seg")
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    try:
        adapter.prepare_segmentation(empty_raw, tmp_path / "out")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for missing raw layout")
