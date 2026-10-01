"""Filesystem helpers shared across the classification and detection adapters."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def link_image(src: Path, dst: Path) -> None:
    """
    Materialize ``src`` at ``dst`` as a real directory entry (hardlink, or a copy).

    Deliberately not a symlink: a symlink here would resolve outside the
    canonical split directory (back into the raw download ``src`` lives in),
    which downstream consumers that validate resolved paths against that
    directory (e.g. Supervision's ``DetectionDataset.from_coco``, which
    rejects any image path escaping the declared images directory) refuse
    to load. A hardlink keeps the old symlink approach's space savings (same
    inode, no duplicated bytes) while still resolving to a real path inside
    the split directory; falls back to a copy only when ``src``/``dst`` are
    on different filesystems (hardlinks can't cross devices).
    """
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
