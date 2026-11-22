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

    for split in ("train", "valid"):
        assert (output_dir / split / "images").is_dir()
        assert (output_dir / split / "masks").is_dir()
    # Official splits: train -> train, val -> valid, and no test (no public masks).
    assert len(list((output_dir / "train" / "images").glob("*.jpg"))) == 6
    assert len(list((output_dir / "valid" / "images").glob("*.jpg"))) == 2
    assert not (output_dir / "test").exists()


def test_prepare_segmentation_pairs_masks_with_images(
    raw_bdd100k_semantic_seg_dir: Path,
) -> None:
    output_dir = raw_bdd100k_semantic_seg_dir.parent / "seg_out"
    adapter = get("bdd100k-semantic-seg")
    adapter.prepare_segmentation(raw_bdd100k_semantic_seg_dir, output_dir)

    for image_path in (output_dir / "valid" / "images").glob("*.jpg"):
        mask_path = output_dir / "valid" / "masks" / f"{image_path.stem}.png"
        assert mask_path.is_file()


def test_prepare_segmentation_preserves_mask_pixel_values(
    raw_bdd100k_semantic_seg_dir: Path,
) -> None:
    output_dir = raw_bdd100k_semantic_seg_dir.parent / "seg_out"
    adapter = get("bdd100k-semantic-seg")
    adapter.prepare_segmentation(raw_bdd100k_semantic_seg_dir, output_dir)

    mask_path = output_dir / "valid" / "masks" / "v0.png"
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
    for mask_path in (output_dir / "train" / "masks").glob("*.png"):
        assert np.asarray(Image.open(mask_path))[0, 0] == 255


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


def test_prepare_segmentation_reads_the_sharded_hub_layout(
    raw_bdd100k_semantic_seg_dir: Path,
) -> None:
    """``data/images_10k/10k`` + ``data/seg_maps/labels`` with shard folders."""
    hub = raw_bdd100k_semantic_seg_dir.parent / "hub"
    for split in ("train", "val"):
        images = hub / "data" / "images_10k" / "10k" / split / "shard_000"
        masks = hub / "data" / "seg_maps" / "labels" / split / "shard_000"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        src = raw_bdd100k_semantic_seg_dir
        for image in (src / "images" / "10k" / split).glob("*.jpg"):
            image.rename(images / image.name)
        for mask in (src / "labels" / "sem_seg" / "masks" / split).glob("*.png"):
            mask.rename(masks / f"{mask.stem}_train_id.png")
    output_dir = hub.parent / "seg_out"
    get("bdd100k-semantic-seg").prepare_segmentation(hub, output_dir)

    assert len(list((output_dir / "train" / "masks").glob("*.png"))) == 6
    assert len(list((output_dir / "valid" / "masks").glob("*.png"))) == 2
