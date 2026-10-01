"""
BDD100K semantic segmentation adapter.

Source: the Berkeley DeepDrive "10K Images" package
(``images/10k/{train,val,test}/*.jpg``) plus the official ``sem_seg`` label
release's **mask** format (``labels/sem_seg/masks/{train,val}/*.png``:
one 1-channel 8-bit PNG per image, pixel value = class id, ``255`` =
"unknown"/ignore). This adapter reads the mask PNGs directly and never
touches the JSON/RLE/polygon label variants BDD100K also ships, so it has
no dependency on ``scalabel`` or the official toolkit.

Note the 10K Images package is a *different* image set from the 100K
Images package the detection/classification adapters use: per BDD100K's
own docs it "is not a subset of the 100K images, even though there is a
significant overlap", so a semantic segmentation run needs its own
download, not the one already used for detection/classification.

BDD100K's own labelled splits are ``train`` (7,000) and ``val`` (1,000).
This adapter keeps ``val`` as the canonical ``test`` split and carves a
seeded validation set out of ``train`` (``_VAL_FRACTION``), the same
convention used by the detection/classification adapters. The public
``test`` split (2,000 images) ships with no masks and is not used here.

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
from bdd100k_toolkit.utils.split import seeded_holdout

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

_VAL_FRACTION = 0.15
_SPLIT_SEED = 42


@register
class BDD100KSemanticSegAdapter(SemanticSegAdapter):
    """Adapter for the BDD100K 19-class semantic segmentation dataset."""

    spec = SemanticSegSpec(
        key="bdd100k-semantic-seg",
        display_name="BDD100K Semantic Segmentation",
        classes=_CLASSES,
        description=(
            "BDD100K's semantic segmentation task: 8,000 labelled 1280x720 "
            "images (7,000 train + 1,000 val, drawn from the separate 10K "
            "Images package) with dense per-pixel labels across 19 "
            "Cityscapes-style classes."
        ),
        homepage="https://www.bdd100k.com/",
        license=(
            "BDD100K License (non-commercial research/education; "
            "registration-gated, no redistribution)."
        ),
    )

    def prepare_segmentation(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert the official BDD100K 10K-images sem_seg release into canonical."""
        image_dir = raw_dir / "images" / "10k"
        mask_dir = raw_dir / "labels" / "sem_seg" / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {raw_dir}/images/10k and {raw_dir}/labels/sem_seg/masks."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_pairs = _pair_images_and_masks(image_dir / "train", mask_dir / "train")
        train_stems, val_stems = seeded_holdout(
            sorted(train_pairs), _VAL_FRACTION, _SPLIT_SEED
        )
        write_image_mask_split(
            {s: train_pairs[s] for s in train_stems}, output_dir / "train"
        )
        write_image_mask_split(
            {s: train_pairs[s] for s in val_stems},
            output_dir / "valid",
            required=False,
        )

        test_pairs = _pair_images_and_masks(image_dir / "val", mask_dir / "val")
        write_image_mask_split(test_pairs, output_dir / "test")


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
    masks_by_stem = {_mask_stem(p): p for p in mask_dir.glob("*.png")}
    pairs: dict[str, tuple[Path, Path]] = {}
    unmatched_images = 0
    for image_path in image_dir.glob("*.jpg"):
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
    *,
    required: bool = True,
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
        required=required,
    )
