"""
BDD100K detection adapter: multi-class autonomous-driving object detection.

Ported from DetectionBench's ``datasets/bdd100k.py`` (same conversion logic,
same class list, same split convention), kept here rather than depended on,
since this project doesn't take DetectionBench as a runtime dependency.

Source: the Berkeley DeepDrive "100k Images" + "Labels" release, as
downloaded via the third-party Kaggle mirror
https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k (uploaded by
Kaggle user SoleSensei, not by Berkeley DeepDrive; the official site was
registration-gated at the time this was fetched; the Kaggle copy carries no
separate license of its own, see the License note below):
``images/100k/{train,val}/**/*.jpg`` (1280x720 JPEGs; this mirror's zip
layout splits the 70,000 train images across four subfolders confusingly
named ``trainA``/``trainB``/``testA``/``testB``; all four are genuinely
part of ``train`` (verified: their combined recursive count is exactly
70,000, matching the official train total, and every one of
``train.json``'s 69,863 entries resolves to a file inside one of these
four subfolders); the loose, separate ``images/100k/test/`` top-level
directory (293 files, no matching labels) is the mirror's copy of the
*actual* unlabelled official test split and is correctly never referenced
by this adapter. ``val`` is flat. All image subfolders are walked
recursively here.) plus either the 2020 revision
``labels/det_20/det_{train,val}.json`` (preferred when present, since the official
docs recommend it) or the original 2018
``labels/bdd100k_labels_images_{train,val}.json`` (one dict per image:
``attributes`` + a ``labels`` list mixing ``box2d`` detection boxes with
``poly2d`` lane/drivable-area annotations). Only ``box2d`` entries are
converted; lane markings and drivable-area polygons are not a detection
task and are dropped; ``other person`` / ``other vehicle`` / ``trailer``
regions (ignored regions in the official evaluation) are dropped *and
counted*, and the 2020 names ``pedestrian/motorcycle/bicycle`` are accepted
as aliases of the 2018 ``person/motor/bike``. Every drop is reported per
split, and an empty ``train``/``test`` split raises. The official ``test``
split (20,000 images) ships with no labels (held out for the leaderboard)
and is not used here.

This is the same raw download the classification adapters
(``bdd100k_toolkit.classification.datasets``) derive their labels from
``attributes.{weather,timeofday,scene}`` on the very same JSON entries,
so a single BDD100K download prepares every task in this project.

BDD100K's own labelled splits are ``train`` (69,863 labelled images) and
``val`` (10,000). This adapter keeps ``val`` as the canonical ``test``
(consistent with how ``val``'s ground truth in this release is only usable
as a held-out eval set, not for training) and carves a seeded validation
set out of ``train`` (``_VAL_FRACTION``).

License: Berkeley DeepDrive's own **BDD100K License**: free for
non-commercial research/educational use; commercial use and redistribution
require separate permission. The data itself is registration-gated (manual
DUA click-through on the official site). That Kaggle mirror declares no
license of its own (Kaggle lists it as "Other (specified in description)",
i.e. the original BDD100K terms still apply) and is not an official
Berkeley DeepDrive distribution channel, so it does not change the
underlying terms or grant any redistribution right. Fine to build/evaluate
against locally. **No Hugging Face mirror**: the DUA does not permit
redistribution, regardless of which download channel the raw files came
from.
"""

from __future__ import annotations

import json
import random
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from bdd100k_toolkit.detection.base import (
    COCO_ANNOTATION_FILENAME,
    DatasetAdapter,
    DatasetSpec,
    link_image,
)
from bdd100k_toolkit.detection.registry import register
from bdd100k_toolkit.utils.checks import format_counts, require_nonempty

_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
]
_CLASS_TO_ID = {name: i for i, name in enumerate(_CLASSES)}

_IMAGE_WIDTH = 1280
_IMAGE_HEIGHT = 720

# Label files, in order of preference. ``det_20`` is the 2020 revision the
# official docs recommend; the 2018 file is deprecated ("kept for comparison
# with legacy results") but still the only one carrying the frame
# ``attributes`` used by the classification tasks.
_LABEL_SOURCES = (
    ("det_20", "det_20/det_{split}.json"),
    ("legacy", "bdd100k_labels_images_{split}.json"),
)

# The 2018 labels use ``person/motor/bike``; the official toolkit's config
# renames them ``pedestrian/motorcycle/bicycle`` and ``det_20`` may use either.
# Canonical names here stay the legacy ones so class ids don't change.
_CATEGORY_ALIASES = {
    "pedestrian": "person",
    "motorcycle": "motor",
    "bicycle": "bike",
}
# Regions the official evaluation treats as "ignored" rather than as classes.
_IGNORED_CATEGORIES = frozenset({"other person", "other vehicle", "trailer"})

_VAL_FRACTION = 0.15
_SPLIT_SEED = 42


@register
class BDD100KDetectionAdapter(DatasetAdapter):
    """Adapter for the BDD100K multi-class driving-scene detection dataset."""

    spec = DatasetSpec(
        key="bdd100k-detection",
        display_name="BDD100K Detection",
        classes=_CLASSES,
        description=(
            "BDD100K is a large-scale, diverse driving-video dataset; this "
            "adapter covers its 100K-image object-detection task: 79,863 "
            "labelled 1280x720 dashcam images (69,863 train + 10,000 val, "
            "the official held-out test split has no released labels) "
            "annotated for 10 classes spanning vulnerable road users "
            "(person, rider), vehicles (car, truck, bus, train, motor, "
            "bike), and traffic control (traffic light, traffic sign), "
            "captured across diverse weather, time-of-day, and scene "
            "conditions in the US."
        ),
        homepage="https://www.bdd100k.com/",
        citation=(
            "@inproceedings{yu2020bdd100k,\n"
            "  title={BDD100K: A Diverse Driving Dataset for Heterogeneous "
            "Multitask Learning},\n"
            "  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, "
            "Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, "
            "Vashisht and Darrell, Trevor},\n"
            "  booktitle={Proceedings of the IEEE/CVF Conference on Computer "
            "Vision and Pattern Recognition},\n"
            "  pages={2636--2645},\n"
            "  year={2020}\n"
            "}"
        ),
        license=(
            "BDD100K License (non-commercial research/education; "
            "registration-gated, no redistribution)."
        ),
    )

    def prepare_coco(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official BDD100K 100k-images release into canonical COCO."""
        image_dir = raw_dir / "images" / "100k"
        label_dir = raw_dir / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/images/100k and {raw_dir}/labels."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        source, train_json, val_json = _resolve_label_files(label_dir)
        print(f"Using {source} box2d labels: {train_json.name}, {val_json.name}")

        train_entries = json.loads(train_json.read_text(encoding="utf-8"))
        train_images_by_path = _index_images(image_dir / "train")
        shuffled = train_entries[:]
        random.Random(_SPLIT_SEED).shuffle(shuffled)  # noqa: S311  # nosec: B311
        n_val = max(1, round(len(shuffled) * _VAL_FRACTION))
        val_names = {e["name"] for e in shuffled[:n_val]}

        _write_split(
            [e for e in train_entries if e["name"] not in val_names],
            train_images_by_path,
            output_dir / "train",
        )
        _write_split(
            [e for e in train_entries if e["name"] in val_names],
            train_images_by_path,
            output_dir / "valid",
            required=False,
        )

        val_entries = json.loads(val_json.read_text(encoding="utf-8"))
        val_images_by_path = _index_images(image_dir / "val")
        _write_split(val_entries, val_images_by_path, output_dir / "test")


def _resolve_label_files(label_dir: Path) -> tuple[str, Path, Path]:
    """Pick one label source whose train *and* val files both exist."""
    tried: list[str] = []
    for source, template in _LABEL_SOURCES:
        train_json = label_dir / template.format(split="train")
        val_json = label_dir / template.format(split="val")
        if train_json.is_file() and val_json.is_file():
            return source, train_json, val_json
        tried.append(f"{train_json} + {val_json}")
    raise FileNotFoundError(
        "No BDD100K detection label files found. Looked for: " + "; ".join(tried)
    )


def _index_images(root: Path) -> dict[str, Path]:
    """Map ``file name -> full path`` for every jpg under ``root`` (recursive)."""
    return {p.name: p for p in root.rglob("*.jpg")}


def _image_size(path: Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` from the image header, or None if unreadable."""
    try:
        with Image.open(path) as image:
            return image.size
    except OSError:
        return None


@dataclass
class _SplitStats:
    """Everything an adapter run drops or counts, reported at the end of a split."""

    entries: int = 0
    images: int = 0
    boxes: int = 0
    crowd_boxes: int = 0
    degenerate_boxes: int = 0
    missing_images: int = 0
    unreadable_images: int = 0
    per_class: Counter[str] = field(default_factory=Counter)
    ignored_regions: Counter[str] = field(default_factory=Counter)
    unknown_categories: Counter[str] = field(default_factory=Counter)


def _write_split(
    entries: list[dict[str, Any]],
    images_by_name: dict[str, Path],
    split_output_dir: Path,
    *,
    required: bool = True,
) -> None:
    """Emit one canonical COCO split from a subset of BDD100K label entries."""
    split_output_dir.mkdir(parents=True, exist_ok=True)
    stats = _SplitStats(entries=len(entries))

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for entry in entries:
        name = entry["name"]
        src = images_by_name.get(name)
        if src is None:
            stats.missing_images += 1
            continue
        link_image(src, split_output_dir / name)
        size = _image_size(src)
        if size is None:
            stats.unreadable_images += 1
            size = (_IMAGE_WIDTH, _IMAGE_HEIGHT)
        image_id = len(images)
        images.append(
            {"id": image_id, "file_name": name, "width": size[0], "height": size[1]}
        )
        for label in entry.get("labels") or []:
            annotation = _label_to_annotation(
                label, image_id, len(annotations) + 1, stats
            )
            if annotation is not None:
                annotations.append(annotation)

    stats.images = len(images)
    stats.boxes = len(annotations)
    _write_coco(split_output_dir, images, annotations)
    _report_split(split_output_dir.name, stats)
    require_nonempty(
        split_output_dir.name,
        stats.images,
        f"{stats.entries} label entries, {stats.missing_images} with no matching "
        "image file (check the raw_dir layout: images/100k/{train,val}/**/*.jpg)",
        required=required,
    )


def _label_to_annotation(
    label: dict[str, Any], image_id: int, annotation_id: int, stats: _SplitStats
) -> dict[str, Any] | None:
    """Convert one raw label to a COCO annotation, counting why it was dropped."""
    box = label.get("box2d")
    if box is None:  # lane / drivable-area polygons: not a detection label
        return None
    category: str | None = label.get("category")
    if category is not None:
        category = _CATEGORY_ALIASES.get(category, category)
    if category in _IGNORED_CATEGORIES:
        stats.ignored_regions[category] += 1
        return None
    if category not in _CLASS_TO_ID:
        stats.unknown_categories[str(category)] += 1
        return None
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        stats.degenerate_boxes += 1
        return None

    attributes = label.get("attributes") or {}
    crowd = bool(attributes.get("crowd", False))
    stats.per_class[category] += 1
    stats.crowd_boxes += crowd
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": _CLASS_TO_ID[category],
        "bbox": [x1, y1, box_width, box_height],
        "area": float(box_width) * float(box_height),
        "segmentation": [],
        "iscrowd": int(crowd),
        "attributes": {
            key: bool(attributes[key])
            for key in ("occluded", "truncated")
            if key in attributes
        },
    }


def _write_coco(
    split_output_dir: Path,
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> None:
    payload = {
        "info": {"description": f"BDD100K canonical COCO ({split_output_dir.name})"},
        "licenses": [
            {
                "id": 1,
                "name": "BDD100K License",
                "url": BDD100KDetectionAdapter.spec.homepage,
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": name, "supercategory": "none"}
            for i, name in enumerate(_CLASSES)
        ],
    }
    (split_output_dir / COCO_ANNOTATION_FILENAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _report_split(split: str, stats: _SplitStats) -> None:
    """Print a per-split summary and warn about anything suspicious."""
    print(
        f"[{split}] {stats.images} images, {stats.boxes} boxes "
        f"({stats.crowd_boxes} crowd); skipped: {stats.missing_images} missing "
        f"images, {stats.degenerate_boxes} degenerate boxes"
    )
    print(f"[{split}] boxes per class: {format_counts(dict(stats.per_class))}")
    if stats.ignored_regions:
        print(
            f"[{split}] ignored regions dropped: "
            f"{format_counts(dict(stats.ignored_regions))}"
        )
    if stats.unreadable_images:
        warnings.warn(
            f"[{split}] {stats.unreadable_images} images had unreadable headers; "
            f"assumed {_IMAGE_WIDTH}x{_IMAGE_HEIGHT}.",
            UserWarning,
            stacklevel=2,
        )
    if stats.unknown_categories:
        warnings.warn(
            f"[{split}] dropped boxes with unknown categories: "
            f"{format_counts(dict(stats.unknown_categories))}",
            UserWarning,
            stacklevel=2,
        )
    empty = [name for name in _CLASSES if stats.per_class[name] == 0]
    if stats.images and empty:
        warnings.warn(
            f"[{split}] no boxes for classes: {', '.join(empty)}",
            UserWarning,
            stacklevel=2,
        )
