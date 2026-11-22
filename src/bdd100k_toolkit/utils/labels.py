"""
Read BDD100K attribute and box labels from either shipped layout.

* a single file per split, ``<label_dir>/bdd100k_labels_images_<split>.json``
  (a list with one entry per image, the 2018 download), or
* one JSON per image under ``<label_dir>/100k/<split>/`` (the ``bdd100k_labels.zip``
  package, which also holds the test labels); the folder may be sharded into
  sub-folders, as on the Hugging Face copy.

``official_dirs`` finds the image and label roots of a download, whether it is the
Berkeley zips unpacked side by side (``images/100k`` + ``labels``) or the Hugging
Face repo ``dronefreak/BDD100K`` (``data/images_100k/100k`` + ``data/labels``).

Both come back as the same list of entries, ``{"name", "attributes", "labels"}``,
so the adapters never see which one they were given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PER_IMAGE_DIR = "100k"


def official_dirs(
    raw_dir: Path, labels_dir: Path | None = None
) -> tuple[Path, Path] | None:
    """Return ``(image_dir, label_dir)`` for an official-style download, or None."""
    for base in (raw_dir, raw_dir / "data"):
        for images in (base / "images" / "100k", base / "images_100k" / "100k"):
            labels = labels_dir or base / "labels"
            if images.is_dir() and labels.is_dir():
                return images, labels
    return None


def label_source(label_dir: Path, split: str) -> Path | None:
    """Return the single-file JSON or the per-image folder for ``split``, if any."""
    single = label_dir / f"bdd100k_labels_images_{split}.json"
    if single.is_file():
        return single
    folder = label_dir / PER_IMAGE_DIR / split
    if folder.is_dir() and next(folder.rglob("*.json"), None) is not None:
        return folder
    return None


def read_entries(source: Path) -> list[dict[str, Any]]:
    """Load the entries of a single-file JSON or of a per-image folder."""
    if source.is_file():
        entries: list[dict[str, Any]] = json.loads(source.read_text(encoding="utf-8"))
        return entries
    entries = []
    for path in sorted(source.rglob("*.json"), key=lambda p: p.name):  # shard-proof
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data["frames"]
        if len(frames) != 1:
            raise ValueError(f"{path}: expected one frame, found {len(frames)}")
        entries.append(
            {
                "name": f"{data['name']}.jpg",
                "attributes": data.get("attributes", {}),
                "labels": frames[0].get("objects", []),
            }
        )
    return entries
