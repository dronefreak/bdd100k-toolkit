"""
Decoder for BDD100K's RGBA "bitmask" label format.

Used by both the instance-segmentation and panoptic-segmentation adapters
(``segmentation/instance/datasets/bdd100k.py``,
``segmentation/panoptic/datasets/bdd100k.py``): both tasks' raw labels
ship in this exact same 4-channel PNG encoding.

Format (documented at
https://doc.bdd100k.com, in doc/source/format.rst's ".. _bitmask:" section
in the official `bdd100k/bdd100k` toolkit, and independently verified here
against that repo's own test fixture,
``tests/label/testcases/bitmasks/quasi-video/insseg_bitmask.png``):

- R: category id, 1-indexed (0 = background / no annotation).
- G: 4 packed instance attribute bits: ``(truncated << 3) + (occluded << 2)
  + (crowd << 1) + ignored``. Instances with ``crowd`` or ``ignored`` set are
  excluded from evaluation.
- B, A: combined into a single per-image instance id via ``(B << 8) + A``
  (so up to 65536 distinct instances per image).

This module only *decodes* that pixel format with plain ``numpy``/``PIL``;
it does not depend on ``scalabel`` or the official toolkit at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_CROWD_BIT = 1
_IGNORED_BIT = 0


@dataclass(frozen=True)
class DecodedInstance:
    """One decoded instance from a bitmask PNG."""

    instance_id: int
    category_id: int
    mask: np.ndarray  # bool array, shape (H, W)
    is_crowd: bool
    is_ignored: bool


def decode_bitmask(path: Path) -> list[DecodedInstance]:
    """Decode a BDD100K RGBA bitmask PNG into a list of per-instance masks."""
    arr = np.asarray(Image.open(path).convert("RGBA"))
    category = arr[..., 0].astype(np.int32)
    attrs = arr[..., 1].astype(np.int32)
    instance_id = (arr[..., 2].astype(np.int32) << 8) + arr[..., 3].astype(np.int32)

    # Background pixels (category 0) never form an instance.
    foreground = category > 0
    combined_key = (category.astype(np.int64) << 32) | instance_id.astype(np.int64)
    combined_key = np.where(foreground, combined_key, -1)

    instances: list[DecodedInstance] = []
    for key in np.unique(combined_key):
        if key < 0:
            continue
        mask = combined_key == key
        # attrs is constant within one instance by construction; sample any pixel.
        ys, xs = np.nonzero(mask)
        sample_attr = int(attrs[ys[0], xs[0]])
        instances.append(
            DecodedInstance(
                instance_id=int(key & 0xFFFFFFFF),
                category_id=int((key >> 32) & 0xFFFFFFFF),
                mask=mask,
                is_crowd=bool((sample_attr >> _CROWD_BIT) & 1),
                is_ignored=bool((sample_attr >> _IGNORED_BIT) & 1),
            )
        )
    return instances
