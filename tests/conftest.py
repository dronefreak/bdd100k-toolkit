"""Shared pytest fixtures for BDD100K-Toolkit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_entry(name: str, weather: str, timeofday: str, scene: str) -> dict:
    return {
        "name": name,
        "attributes": {"weather": weather, "timeofday": timeofday, "scene": scene},
        "labels": [],
    }


def _make_detection_entry(name: str, boxes: list[tuple[str, tuple]]) -> dict:
    return {
        "name": name,
        "attributes": {},
        "labels": [
            {
                "category": category,
                "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
            for category, (x1, y1, x2, y2) in boxes
        ],
    }


@pytest.fixture
def raw_bdd100k_dir(tmp_path: Path) -> Path:
    """Build a minimal fake BDD100K raw download (images + native label JSON)."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "images" / "100k" / "train").mkdir(parents=True)
    (raw_dir / "images" / "100k" / "val").mkdir(parents=True)
    (raw_dir / "labels").mkdir(parents=True)

    train_entries = [
        _make_entry("t0.jpg", "clear", "daytime", "city street"),
        _make_entry("t1.jpg", "rainy", "night", "highway"),
        _make_entry("t2.jpg", "snowy", "dawn/dusk", "tunnel"),
        _make_entry("t3.jpg", "clear", "daytime", "residential"),
        _make_entry("t4.jpg", "undefined", "undefined", "undefined"),  # dropped
        _make_entry("t5.jpg", "overcast", "daytime", "parking lot"),
    ]
    val_entries = [
        _make_entry("v0.jpg", "foggy", "night", "gas stations"),
        _make_entry("v1.jpg", "clear", "daytime", "city street"),
    ]

    for entry in train_entries:
        (raw_dir / "images" / "100k" / "train" / entry["name"]).write_bytes(b"fake")
    for entry in val_entries:
        (raw_dir / "images" / "100k" / "val" / entry["name"]).write_bytes(b"fake")

    (raw_dir / "labels" / "bdd100k_labels_images_train.json").write_text(
        json.dumps(train_entries)
    )
    (raw_dir / "labels" / "bdd100k_labels_images_val.json").write_text(
        json.dumps(val_entries)
    )
    return raw_dir


@pytest.fixture
def raw_bdd100k_semantic_seg_dir(tmp_path: Path) -> Path:
    """Build a minimal fake BDD100K raw download for semantic segmentation."""
    import numpy as np
    from PIL import Image

    raw_dir = tmp_path / "raw"
    (raw_dir / "images" / "10k" / "train").mkdir(parents=True)
    (raw_dir / "images" / "10k" / "val").mkdir(parents=True)
    (raw_dir / "labels" / "sem_seg" / "masks" / "train").mkdir(parents=True)
    (raw_dir / "labels" / "sem_seg" / "masks" / "val").mkdir(parents=True)

    train_names = [f"t{i}" for i in range(6)]
    val_names = [f"v{i}" for i in range(2)]

    for i, name in enumerate(train_names):
        Image.new("RGB", (8, 8), color=(i, i, i)).save(
            raw_dir / "images" / "10k" / "train" / f"{name}.jpg"
        )
        mask = np.full((8, 8), i % 19, dtype=np.uint8)
        mask[0, 0] = 255  # ignore pixel
        Image.fromarray(mask).save(
            raw_dir / "labels" / "sem_seg" / "masks" / "train" / f"{name}.png"
        )
    for i, name in enumerate(val_names):
        Image.new("RGB", (8, 8), color=(i, i, i)).save(
            raw_dir / "images" / "10k" / "val" / f"{name}.jpg"
        )
        mask = np.full((8, 8), i % 19, dtype=np.uint8)
        Image.fromarray(mask).save(
            raw_dir / "labels" / "sem_seg" / "masks" / "val" / f"{name}.png"
        )
    return raw_dir


@pytest.fixture
def raw_bdd100k_detection_dir(tmp_path: Path) -> Path:
    """Build a minimal fake BDD100K raw download with box2d detection labels."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "images" / "100k" / "train").mkdir(parents=True)
    (raw_dir / "images" / "100k" / "val").mkdir(parents=True)
    (raw_dir / "labels").mkdir(parents=True)

    train_entries = [
        _make_detection_entry(
            "t0.jpg", [("car", (10, 10, 50, 50)), ("person", (60, 60, 80, 100))]
        ),
        _make_detection_entry("t1.jpg", [("truck", (0, 0, 100, 100))]),
        _make_detection_entry("t2.jpg", [("rider", (5, 5, 20, 40))]),
        _make_detection_entry("t3.jpg", []),
        _make_detection_entry("t4.jpg", [("bike", (1, 1, 30, 30))]),
        _make_detection_entry(
            "t5.jpg", [("traffic light", (1, 1, 5, 5)), ("bad-category", (1, 1, 5, 5))]
        ),
    ]
    val_entries = [
        _make_detection_entry("v0.jpg", [("bus", (0, 0, 40, 40))]),
        _make_detection_entry("v1.jpg", [("motor", (0, 0, 20, 20))]),
    ]

    for entry in train_entries:
        (raw_dir / "images" / "100k" / "train" / entry["name"]).write_bytes(b"fake")
    for entry in val_entries:
        (raw_dir / "images" / "100k" / "val" / entry["name"]).write_bytes(b"fake")

    (raw_dir / "labels" / "bdd100k_labels_images_train.json").write_text(
        json.dumps(train_entries)
    )
    (raw_dir / "labels" / "bdd100k_labels_images_val.json").write_text(
        json.dumps(val_entries)
    )
    return raw_dir


_TINY_COLORS = {"red": (220, 20, 20), "green": (20, 200, 20), "blue": (20, 20, 220)}


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Three classes of flat colour images in canonical train/valid/test folders."""
    from PIL import Image

    sizes = {"train": 12, "valid": 4, "test": 4}
    for split, count in sizes.items():
        for name, color in _TINY_COLORS.items():
            folder = tmp_path / split / name
            folder.mkdir(parents=True)
            for i in range(count):
                jitter = tuple(min(255, c + i) for c in color)
                Image.new("RGB", (64, 36), jitter).save(folder / f"{i}.jpg")
    return tmp_path
