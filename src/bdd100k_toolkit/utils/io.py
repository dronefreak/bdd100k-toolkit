"""Filesystem and image helpers shared across the adapters."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from rich.progress import track

from bdd100k_toolkit.utils.console import RichConsoleManager


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


# --- optional shrinking of images while preparing a dataset -------------------

DEFAULT_JPEG_QUALITY = 90
MIN_MAX_WIDTH = 32
MAX_JPEG_QUALITY = 100
_MIN_PARALLEL_JOBS = 2


@dataclass(frozen=True)
class ImageOptions:
    """
    How prepared images are written.

    BDD100K images are 1280x720, and a model trained at 224 px never needs that
    much. Decoding and augmenting full-size JPEGs is what limits training speed
    (about 1,100 img/s with Ultralytics on the reference machine, against 2,700
    to 4,000 img/s on 512 px wide copies, with no accuracy loss that showed up
    in a 6-epoch check). Shrunk copies are real files, not hardlinks.

    Attributes:
        max_width: Images wider than this are resized to this width, keeping the
            aspect ratio; narrower ones are hardlinked untouched. None keeps
            every image as it is (hardlinks).
        shrink_test: Also shrink the ``test`` split. Off by default so the
            reported numbers stay on the original-resolution test images.
        jpeg_quality: Quality of the resized JPEGs.
        workers: Processes used for resizing (default: all CPU cores).

    """

    max_width: int | None = None
    shrink_test: bool = False
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    workers: int | None = None

    def __post_init__(self) -> None:
        """Validate the options."""
        if self.max_width is not None and self.max_width < MIN_MAX_WIDTH:
            raise ValueError(
                f"max_width must be at least {MIN_MAX_WIDTH}, got {self.max_width}"
            )
        if not 1 <= self.jpeg_quality <= MAX_JPEG_QUALITY:
            raise ValueError(f"jpeg_quality must be 1..100, got {self.jpeg_quality}")
        if self.workers is not None and self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}")

    def shrinks(self, split: str) -> bool:
        """Return True when images of ``split`` are written resized."""
        return self.max_width is not None and (split != "test" or self.shrink_test)


def resize_image(src: Path, dst: Path, max_width: int, quality: int) -> str:
    """
    Write ``src`` to ``dst`` no wider than ``max_width``; return what was done.

    Returns ``"resized"``, ``"linked"`` (already narrow enough, so hardlinked) or
    ``"skipped"`` (``dst`` already exists). The file is written to a ``.part``
    name first and renamed, so an interrupted run never leaves a half-written
    ``.jpg`` behind.
    """
    if dst.exists():
        return "skipped"
    try:
        with Image.open(src) as image:
            width, height = image.size
            if width <= max_width:
                link_image(src, dst)
                return "linked"
            new_height = max(1, round(height * max_width / width))
            image.draft("RGB", (max_width, new_height))  # fast reduced-size decode
            resized = image.convert("RGB").resize(
                (max_width, new_height), Image.Resampling.LANCZOS
            )
        part = dst.with_name(dst.name + ".part")
        resized.save(part, "JPEG", quality=quality)
        os.replace(part, dst)
    except OSError as err:
        raise RuntimeError(f"Failed to resize {src}: {err}") from err
    return "resized"


def _resize_job(job: tuple[Path, Path, int, int]) -> str:
    return resize_image(*job)


def materialize_images(
    jobs: list[tuple[Path, Path]], options: ImageOptions, *, shrink: bool
) -> Counter[str]:
    """
    Write every ``(src, dst)`` pair, hardlinking or (if ``shrink``) resizing.

    Resizing runs in parallel and shows a progress bar; hardlinking is fast
    enough to stay serial. Returns how many images were ``resized``, ``linked``
    or ``skipped`` (destination already existed).
    """
    counts: Counter[str] = Counter()
    if not shrink or options.max_width is None:
        for src, dst in jobs:
            counts["skipped" if dst.exists() else "linked"] += 1
            link_image(src, dst)
        return counts

    work = [(src, dst, options.max_width, options.jpeg_quality) for src, dst in jobs]
    workers = options.workers or os.cpu_count() or 1
    parallel = workers > 1 and len(work) >= _MIN_PARALLEL_JOBS
    executor = ProcessPoolExecutor(max_workers=workers) if parallel else None
    results: Iterable[str] = (
        executor.map(_resize_job, work, chunksize=64)
        if executor is not None
        else map(_resize_job, work)
    )
    try:
        for outcome in track(
            results,
            total=len(work),
            description=f"resizing to {options.max_width}px wide",
            console=RichConsoleManager.get_console(),
        ):
            counts[outcome] += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return counts
