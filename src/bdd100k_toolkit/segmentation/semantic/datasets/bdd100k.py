"""
BDD100K semantic segmentation adapter.

Source: the Berkeley DeepDrive "10K Images" package plus the ``sem_seg`` label
masks: one 1-channel 8-bit PNG per image, pixel value = class id, ``255`` =
"unknown"/ignore. This adapter reads the mask PNGs directly and never touches the
JSON/RLE/polygon label variants BDD100K also ships, so it has no dependency on
``scalabel`` or the official toolkit. Three layouts are found under ``raw_dir``
(or ``raw_dir/data``), and the files may sit in ``shard_NNN/`` sub-folders:

* ``images_10k/10k/<split>`` + ``seg_maps/labels/<split>``: the Berkeley zips
  ``bdd100k_images_10k.zip`` + ``bdd100k_seg_maps.zip``, and the Hugging Face copy
  ``dronefreak/BDD100K-Semantic-Segmentation``;
* ``images/10k/<split>`` + ``labels/sem_seg/masks/<split>``: the later label release;
* ``seg/images/<split>`` + ``seg/labels/<split>``: the older ``bdd100k_seg.zip``.

Note the 10K Images package is a *different* image set from the 100K
Images package the detection/classification adapters use: per BDD100K's
own docs it "is not a subset of the 100K images, even though there is a
significant overlap", so a semantic segmentation run needs its own
download, not the one already used for detection/classification.

The splits are the official ones, as for detection and classification: ``train``
(7,000) -> ``train`` and ``val`` (1,000) -> ``valid``. The ``test`` split (2,000
images) ships without masks, so no ``test`` folder is written.

19 classes are evaluated (see ``_CLASSES``); mask value 255 means "ignore"
and is preserved verbatim in the canonical output rather than remapped.

License: Berkeley DeepDrive's own **BDD100K License**; see
``bdd100k_toolkit.detection.datasets.bdd100k`` for the full license note
(identical terms apply to every BDD100K release).
"""

from __future__ import annotations

from pathlib import Path

from bdd100k_toolkit.segmentation.semantic.base import (
    SemanticSegAdapter,
    SemanticSegSpec,
)
from bdd100k_toolkit.segmentation.semantic.registry import register
from bdd100k_toolkit.utils.checks import require_nonempty
from bdd100k_toolkit.utils.io import link_image

_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

# The 7 Cityscapes categories (cityscapesscripts ``labels.py``), used for category IoU.
_CATEGORIES = {
    "flat": ["road", "sidewalk"],
    "construction": ["building", "wall", "fence"],
    "object": ["pole", "traffic light", "traffic sign"],
    "nature": ["vegetation", "terrain"],
    "sky": ["sky"],
    "human": ["person", "rider"],
    "vehicle": ["car", "truck", "bus", "train", "motorcycle", "bicycle"],
}

# (image root, mask root) pairs, relative to ``raw_dir`` or ``raw_dir/data``.
_LAYOUTS = (
    ("images_10k/10k", "seg_maps/labels"),
    ("images/10k", "labels/sem_seg/masks"),
    ("seg/images", "seg/labels"),
)


def find_layout(raw_dir: Path) -> tuple[Path, Path] | None:
    """Return ``(image_root, mask_root)`` of a 10K sem_seg download, or None."""
    for base in (raw_dir, raw_dir / "data"):
        for images, masks in _LAYOUTS:
            if (base / images).is_dir() and (base / masks).is_dir():
                return base / images, base / masks
    return None


@register
class BDD100KSemanticSegAdapter(SemanticSegAdapter):
    """Adapter for the BDD100K 19-class semantic segmentation dataset."""

    spec = SemanticSegSpec(
        key="bdd100k-semantic-seg",
        display_name="BDD100K Semantic Segmentation",
        classes=_CLASSES,
        categories=_CATEGORIES,
        description=(
            "BDD100K's semantic segmentation task: 8,000 labelled 1280x720 "
            "images (7,000 train + 1,000 val, drawn from the separate 10K "
            "Images package) with dense per-pixel labels across 19 "
            "Cityscapes-style classes."
        ),
        homepage="https://www.bdd100k.com/",
        license=(
            "BDD100K License (UC Regents): educational, research and "
            "not-for-profit use and redistribution with the copyright notice "
            "carried forward; commercial use needs separate permission."
        ),
    )

    def prepare_segmentation(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official BDD100K 10K-images sem_seg release into canonical."""
        found = find_layout(raw_dir)
        if found is None:
            expected = ", ".join(f"{i} + {m}" for i, m in _LAYOUTS)
            raise FileNotFoundError(
                f"No 10K images and sem_seg masks under {raw_dir} (or {raw_dir}/data)."
                f" Expected one of: {expected}."
            )
        image_dir, mask_dir = found
        output_dir.mkdir(parents=True, exist_ok=True)

        # Official splits: train -> train, val -> valid; test has no masks.
        for raw_split, split in (("train", "train"), ("val", "valid")):
            pairs = _pair_images_and_masks(image_dir / raw_split, mask_dir / raw_split)
            write_image_mask_split(pairs, output_dir / split)


# The older BDD100K release names masks ``<stem>_train_id.png``; the current
# one uses ``<stem>.png``. Accept both so the pairing doesn't silently find 0.
_MASK_SUFFIXES = ("_train_id",)


def _mask_stem(path: Path) -> str:
    stem = path.stem
    for suffix in _MASK_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _pair_images_and_masks(
    image_dir: Path, mask_dir: Path
) -> dict[str, tuple[Path, Path]]:
    """Map ``file stem -> (image_path, mask_path)`` for every matched pair."""
    masks_by_stem = {_mask_stem(p): p for p in mask_dir.rglob("*.png")}
    pairs: dict[str, tuple[Path, Path]] = {}
    unmatched_images = 0
    for image_path in image_dir.rglob("*.jpg"):
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            unmatched_images += 1
        else:
            pairs[image_path.stem] = (image_path, mask_path)
    unmatched_masks = len(masks_by_stem) - len(pairs)
    if unmatched_images or unmatched_masks:
        print(
            f"[{image_dir.name}] {unmatched_images} images without a mask, "
            f"{unmatched_masks} masks without an image"
        )
    require_nonempty(
        image_dir.name,
        len(pairs),
        f"no image in {image_dir} matches a mask in {mask_dir} by file stem",
    )
    return pairs


def write_image_mask_split(
    pairs: dict[str, tuple[Path, Path]],
    split_output_dir: Path,
) -> None:
    """Materialize an ``images/`` + ``masks/`` canonical split from matched pairs."""
    images_out = split_output_dir / "images"
    masks_out = split_output_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    for stem, (image_path, mask_path) in pairs.items():
        link_image(image_path, images_out / f"{stem}.jpg")
        link_image(mask_path, masks_out / f"{stem}.png")
    print(f"[{split_output_dir.name}] {len(pairs)} image/mask pairs")
    require_nonempty(
        split_output_dir.name,
        len(pairs),
        "no image/mask pairs were written",
    )
