"""
Shared logic for the BDD100K image-attribute classification tasks.

These are *unofficial* tasks: BDD100K itself only ships per-image
``attributes: {weather, timeofday, scene}`` inside its detection label JSON.
The tasks follow three Kaggle re-exports (period / weather / scenario
classification, folders ``train/<class>/*.jpg``) whose labels, checked image by
image against the official JSON, are identical: ``unknown`` is the official
``undefined`` value, and ``dawn/dusk`` is spelled ``dawn or dusk`` because a
slash cannot be a folder name. Their ``test`` folder is the official unlabeled
test split and is never used.

Three input layouts are accepted (auto-detected under ``raw_dir``):

``official``
    ``images/100k/{train,val}/**/*.jpg`` plus
    ``labels/bdd100k_labels_images_{train,val}.json`` (``labels_dir`` may point
    elsewhere, since the images and labels ship as separate archives).
``folders``
    ``{train,val}/<class>/*.jpg``, the Kaggle layout. A loose ``test/`` folder
    is ignored.
``hf``
    The Hugging Face datasets ``dronefreak/BDD100K-{Weather,Period,Scenario}-
    Classification``: ``data/images/{train,valid}/shard_*/`` with the jpgs and a
    ``metadata.jsonl`` per shard (``{"file_name": ..., "label": <class>}``).
    ``raw_dir`` is the downloaded repo root; ``valid`` is the official val set.

Whichever is used, the output and the splits are the same: official ``val``
becomes ``test``, and a seeded 15% of ``train`` becomes ``valid``. The holdout
is chosen over the sorted image names of *all* train images (before any class
filtering), so it is identical across all layouts, across the three tasks, and
for the detection task, and it does not change with ``include_unknown``.

``unknown`` images are kept as a class by default (the Kaggle definition) and
dropped with ``include_unknown=False``.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bdd100k_toolkit.utils.io import ImageOptions, materialize_images
from bdd100k_toolkit.utils.checks import format_counts, require_nonempty
from bdd100k_toolkit.utils.split import seeded_holdout

VAL_FRACTION = 0.15
SPLIT_SEED = 42

UNKNOWN_CLASS = "unknown"
RAW_UNDEFINED = "undefined"

Layout = Literal["official", "folders", "hf"]
INFO_FILENAME = "prepare_info.json"


@dataclass
class _Sample:
    """One train/val image: its class, or the reason it will not be written."""

    name: str
    path: Path | None  # None when the label has no image file
    label: str | None  # class folder; None means "not written"
    reason: str | None = None


def index_images(root: Path) -> dict[str, Path]:
    """Map ``file name -> full path`` for every jpg under ``root`` (recursive)."""
    return {p.name: p for p in root.rglob("*.jpg")}


def load_entries(label_dir: Path, split: str) -> list[dict[str, Any]]:
    """Load one native BDD100K label JSON (``train`` or ``val``)."""
    path = label_dir / f"bdd100k_labels_images_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def detect_layout(raw_dir: Path, labels_dir: Path | None = None) -> Layout:
    """Return which input layout ``raw_dir`` has (``official`` wins if both)."""
    official = (raw_dir / "images" / "100k").is_dir() and (
        labels_dir or raw_dir / "labels"
    ).is_dir()
    folders = (raw_dir / "train").is_dir() and (raw_dir / "val").is_dir()
    hf = (raw_dir / "data" / "images" / "train").is_dir()
    if official:
        return "official"
    if folders:
        return "folders"
    if hf:
        return "hf"
    raise FileNotFoundError(
        f"Unrecognised layout under {raw_dir}. Expected either "
        "images/100k/{train,val} + labels/ (or --labels-dir), "
        "{train,val}/<class>/*.jpg, or the Hugging Face data/images/{train,valid}/."
    )


def prepare_attribute_classification(  # noqa: PLR0913
    raw_dir: Path,
    output_dir: Path,
    attribute: str,
    classes: list[str],
    *,
    aliases: dict[str, str] | None = None,
    include_unknown: bool = True,
    labels_dir: Path | None = None,
    image_options: ImageOptions | None = None,
) -> None:
    """
    Convert a BDD100K download into canonical per-``attribute`` splits.

    Args:
        raw_dir: Download root (see the module docstring for both layouts).
        output_dir: Canonical output root; written as
            ``output_dir/{train,valid,test}/<class>/*.jpg``.
        attribute: One of BDD100K's per-image attribute keys
            (``"weather"``, ``"timeofday"``, ``"scene"``).
        classes: The known class names, excluding ``unknown``. These are the
            output folder names.
        aliases: Raw attribute value -> class name, where they differ
            (``{"dawn/dusk": "dawn or dusk"}``).
        include_unknown: Keep the ``unknown`` (official ``undefined``) class.
        labels_dir: Folder holding the label JSON files (official layout
            only); defaults to ``raw_dir/labels``.
        image_options: Optional shrinking of the written images (see
            :class:`~bdd100k_toolkit.utils.io.ImageOptions`); by default every
            image is hardlinked untouched.

    """
    allowed = [*classes, UNKNOWN_CLASS] if include_unknown else list(classes)
    layout = detect_layout(raw_dir, labels_dir)
    print(f"Input layout: {layout}")
    if layout == "official":
        train, val = _samples_from_official(
            raw_dir,
            labels_dir or raw_dir / "labels",
            attribute,
            classes,
            aliases or {},
            include_unknown,
        )
    elif layout == "hf":
        train, val = _samples_from_hf(raw_dir, classes, include_unknown)
    else:
        train, val = _samples_from_folders(raw_dir, classes, include_unknown)
    options = image_options or ImageOptions()
    _warn_if_not_empty(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, valid_names = seeded_holdout(
        sorted(s.name for s in train), VAL_FRACTION, SPLIT_SEED
    )
    valid_set = set(valid_names)
    split_counts = {
        "train": _write_split(
            [s for s in train if s.name not in valid_set],
            output_dir / "train",
            allowed,
            options,
        ),
        "valid": _write_split(
            [s for s in train if s.name in valid_set],
            output_dir / "valid",
            allowed,
            options,
            required=False,
        ),
        "test": _write_split(val, output_dir / "test", allowed, options),
    }
    _write_info(
        output_dir,
        {
            "attribute": attribute,
            "layout": layout,
            "include_unknown": include_unknown,
            "max_width": options.max_width,
            "shrink_test": options.shrink_test,
            "jpeg_quality": options.jpeg_quality if options.max_width else None,
            "images_per_split": {k: sum(v.values()) for k, v in split_counts.items()},
            "images_per_class": split_counts,
        },
    )


def _warn_if_not_empty(output_dir: Path) -> None:
    """Warn that files already in ``output_dir`` are kept, not overwritten."""
    if output_dir.is_dir() and any(output_dir.iterdir()):
        warnings.warn(
            f"{output_dir} is not empty: existing images are kept as they are "
            "(e.g. an earlier hardlink is not replaced by a resized copy). "
            "Use a fresh --output-dir for a clean result.",
            UserWarning,
            stacklevel=3,
        )


def _write_info(output_dir: Path, info: dict[str, Any]) -> None:
    """Record how the dataset was prepared, so it can be told apart later."""
    (output_dir / INFO_FILENAME).write_text(json.dumps(info, indent=2) + "\n")


def _label_for(
    raw: str | None, classes: list[str], aliases: dict[str, str], include_unknown: bool
) -> tuple[str | None, str | None]:
    """Map a raw attribute value to ``(class, None)`` or ``(None, reason)``."""
    if raw == RAW_UNDEFINED:
        if include_unknown:
            return UNKNOWN_CLASS, None
        return None, "unknown excluded"
    label = aliases.get(raw, raw) if raw is not None else None
    if label in classes:
        return label, None
    return None, f"unexpected value {raw!r}"


def _samples_from_official(  # noqa: PLR0913, PLR0917
    raw_dir: Path,
    label_dir: Path,
    attribute: str,
    classes: list[str],
    aliases: dict[str, str],
    include_unknown: bool,
) -> tuple[list[_Sample], list[_Sample]]:
    """Read train and val samples from the raw images + official label JSON."""
    image_dir = raw_dir / "images" / "100k"
    result: list[list[_Sample]] = []
    for split in ("train", "val"):
        images = index_images(image_dir / split)
        samples: list[_Sample] = []
        for entry in load_entries(label_dir, split):
            src = images.get(entry["name"])
            raw = entry.get("attributes", {}).get(attribute)
            label, reason = _label_for(raw, classes, aliases, include_unknown)
            if src is None:
                label, reason = None, "label without image file"
            samples.append(_Sample(entry["name"], src, label, reason))
        result.append(samples)
    return result[0], result[1]


def _samples_from_folders(
    raw_dir: Path, classes: list[str], include_unknown: bool
) -> tuple[list[_Sample], list[_Sample]]:
    """Read train and val samples from ``{train,val}/<class>/*.jpg`` folders."""
    known = {*classes, UNKNOWN_CLASS}
    result: list[list[_Sample]] = []
    for split in ("train", "val"):
        class_dirs = sorted(p for p in (raw_dir / split).iterdir() if p.is_dir())
        found = {p.name for p in class_dirs}
        if not found <= known:
            raise ValueError(
                f"{raw_dir / split} has class folders {sorted(found - known)} that "
                f"this dataset does not define. Expected a subset of "
                f"{sorted(known)}; is --dataset the right task for this folder?"
            )
        if known - found:
            warnings.warn(
                f"[{split}] no folders for classes: {', '.join(sorted(known - found))}",
                UserWarning,
                stacklevel=2,
            )
        samples: list[_Sample] = []
        for class_dir in class_dirs:
            excluded = class_dir.name == UNKNOWN_CLASS and not include_unknown
            for path in sorted(class_dir.glob("*.jpg")):
                samples.append(
                    _Sample(
                        path.name,
                        path,
                        None if excluded else class_dir.name,
                        "unknown excluded" if excluded else None,
                    )
                )
        result.append(samples)
    test_dir = raw_dir / "test"
    if test_dir.is_dir():
        print(f"Ignoring unlabeled test images in {test_dir}")
    return result[0], result[1]


def _samples_from_hf(
    raw_dir: Path, classes: list[str], include_unknown: bool
) -> tuple[list[_Sample], list[_Sample]]:
    """Read train and val samples from the Hugging Face shards + ``metadata.jsonl``."""
    known = {*classes, UNKNOWN_CLASS}
    result: list[list[_Sample]] = []
    for split in ("train", "valid"):
        samples: list[_Sample] = []
        shards = sorted((raw_dir / "data" / "images" / split).glob("shard_*"))
        for shard in shards:
            for line in (
                (shard / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            ):
                row = json.loads(line)
                name, raw = row["file_name"], row["label"]
                path = shard / name
                if not path.is_file():
                    samples.append(
                        _Sample(name, None, None, "label without image file")
                    )
                elif raw == UNKNOWN_CLASS and not include_unknown:
                    samples.append(_Sample(name, path, None, "unknown excluded"))
                elif raw in known:
                    samples.append(_Sample(name, path, raw))
                else:
                    samples.append(
                        _Sample(name, path, None, f"unexpected value {raw!r}")
                    )
        result.append(samples)
    return result[0], result[1]


def _write_split(
    samples: list[_Sample],
    split_output_dir: Path,
    classes: list[str],
    options: ImageOptions,
    *,
    required: bool = True,
) -> dict[str, int]:
    """Emit one canonical ``<class>/<image>.jpg`` split; return images per class."""
    counts: dict[str, int] = dict.fromkeys(classes, 0)
    dropped: Counter[str] = Counter()
    jobs: list[tuple[Path, Path]] = []
    for sample in samples:
        if sample.label is None or sample.path is None:
            dropped[sample.reason or "no label"] += 1
            continue
        class_dir = split_output_dir / sample.label
        class_dir.mkdir(parents=True, exist_ok=True)
        jobs.append((sample.path, class_dir / sample.name))
        counts[sample.label] += 1
    shrink = options.shrinks(split_output_dir.name)
    written = materialize_images(jobs, options, shrink=shrink)

    total = sum(counts.values())
    how = f"; {format_counts(dict(written))}" if shrink else ""
    print(
        f"[{split_output_dir.name}] {total} images ({format_counts(counts)}); "
        f"dropped: {format_counts(dict(dropped))}{how}"
    )
    problems = {
        reason: count
        for reason, count in dropped.items()
        if reason.startswith(("unexpected", "label without"))
    }
    if problems:
        warnings.warn(
            f"[{split_output_dir.name}] dropped images: {format_counts(problems)}",
            UserWarning,
            stacklevel=2,
        )
    empty = [name for name, count in counts.items() if count == 0]
    if total and empty and split_output_dir.name == "train":
        warnings.warn(
            f"[train] no images for classes: {', '.join(empty)}",
            UserWarning,
            stacklevel=2,
        )
    require_nonempty(
        split_output_dir.name,
        total,
        f"{len(samples)} images considered, dropped: {format_counts(dict(dropped))} "
        "(check the layout: images/100k/{train,val}/**/*.jpg plus labels, or "
        "{train,val}/<class>/*.jpg)",
        required=required,
    )
    return counts
