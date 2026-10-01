"""
Shared logic for deriving BDD100K image-attribute classification tasks.

BDD100K's own object-detection label release,
``labels/bdd100k_labels_images_{train,val}.json``, already tags every image
with ``attributes: {weather, timeofday, scene}``. The Kaggle datasets this
project targets (period/weather/scenario classification, uploaded by
marquis03) are third-party re-exports of exactly these three attributes as
per-class image folders; deriving the same splits directly from the official
label JSON sidesteps needing the Kaggle mirror at all, reuses the raw
download DetectionBench's own ``bdd100k`` detection adapter already expects
(``images/100k/{train,val}/`` + ``labels/``), and avoids depending on a
third party's undocumented class list or split.

Layout expected under ``raw_dir`` (identical to DetectionBench's BDD100K
adapter): ``images/100k/{train,val}/**/*.jpg`` plus
``labels/bdd100k_labels_images_{train,val}.json``.

``undefined`` attribute values (present in a small minority of images, e.g.
rare sensor/annotation gaps) are dropped rather than treated as a class;
they aren't a decidable label, matching how the detection adapter drops
non-``box2d`` labels rather than inventing a placeholder class for them.

BDD100K's own labelled splits are ``train`` (69,863 images) and ``val``
(10,000). As with the detection adapter, ``val`` becomes the canonical
``test`` split, and a seeded fraction of ``train`` is carved out as
``valid``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from bdd100k_toolkit.classification.base import link_image
from bdd100k_toolkit.utils.checks import format_counts, require_nonempty

VAL_FRACTION = 0.15
SPLIT_SEED = 42


def index_images(root: Path) -> dict[str, Path]:
    """Map ``file name -> full path`` for every jpg under ``root`` (recursive)."""
    return {p.name: p for p in root.rglob("*.jpg")}


def load_entries(label_dir: Path, split: str) -> list[dict[str, Any]]:
    """Load one native BDD100K label JSON (``train`` or ``val``)."""
    path = label_dir / f"bdd100k_labels_images_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_attribute_classification(
    raw_dir: Path,
    output_dir: Path,
    attribute: str,
    classes: list[str],
) -> None:
    """
    Convert BDD100K's native release into canonical per-``attribute`` splits.

    Args:
        raw_dir: BDD100K raw download root (``images/100k/`` + ``labels/``).
        output_dir: Canonical output root; written as
            ``output_dir/{train,valid,test}/<class>/*.jpg``.
        attribute: One of BDD100K's per-image attribute keys
            (``"weather"``, ``"timeofday"``, ``"scene"``).
        classes: The attribute's known values, excluding ``"undefined"``.

    """
    image_dir = raw_dir / "images" / "100k"
    label_dir = raw_dir / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected {raw_dir}/images/100k and {raw_dir}/labels.")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_entries = load_entries(label_dir, "train")
    train_images = index_images(image_dir / "train")
    shuffled = train_entries[:]
    random.Random(SPLIT_SEED).shuffle(shuffled)  # noqa: S311  # nosec: B311
    n_val = max(1, round(len(shuffled) * VAL_FRACTION))
    val_names = {e["name"] for e in shuffled[:n_val]}

    _write_split(
        [e for e in train_entries if e["name"] not in val_names],
        train_images,
        output_dir / "train",
        attribute,
        classes,
    )
    _write_split(
        [e for e in train_entries if e["name"] in val_names],
        train_images,
        output_dir / "valid",
        attribute,
        classes,
        required=False,
    )

    val_entries = load_entries(label_dir, "val")
    val_images = index_images(image_dir / "val")
    _write_split(val_entries, val_images, output_dir / "test", attribute, classes)


def _write_split(  # noqa: PLR0913
    entries: list[dict[str, Any]],
    images_by_name: dict[str, Path],
    split_output_dir: Path,
    attribute: str,
    classes: list[str],
    *,
    required: bool = True,
) -> None:
    """Emit one canonical ``<class>/<image>.jpg`` split from label entries."""
    class_set = set(classes)
    counts: dict[str, int] = dict.fromkeys(classes, 0)
    skipped_undefined = 0
    skipped_missing_image = 0

    for entry in entries:
        label = entry.get("attributes", {}).get(attribute)
        if label not in class_set:
            skipped_undefined += 1
            continue
        src = images_by_name.get(entry["name"])
        if src is None:
            skipped_missing_image += 1
            continue
        class_dir = split_output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        link_image(src, class_dir / entry["name"])
        counts[label] += 1

    total = sum(counts.values())
    print(
        f"[{split_output_dir.name}] {total} images ({format_counts(counts)}); "
        f"skipped {skipped_undefined} undefined, {skipped_missing_image} missing image"
    )
    require_nonempty(
        split_output_dir.name,
        total,
        f"{len(entries)} label entries, {skipped_undefined} with an undefined "
        f"'{attribute}' attribute, {skipped_missing_image} with no matching image "
        "file (check the raw_dir layout: images/100k/{train,val}/**/*.jpg)",
        required=required,
    )
