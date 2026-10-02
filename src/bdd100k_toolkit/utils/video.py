"""Reading and writing the videos a demo runs on (OpenCV comes with Ultralytics)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".flv",
    ".wmv",
)


@contextmanager
def open_video(path: Path) -> Iterator[tuple[float, int, Iterator[Image.Image]]]:
    """
    Yield ``(fps, frame_count, frames)``; frames are RGB PIL images.

    ``frame_count`` is 0 when the container does not report a length.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    def frames() -> Iterator[Image.Image]:
        while True:
            ok, bgr = capture.read()
            if not ok:
                return
            yield Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    try:
        yield (
            capture.get(cv2.CAP_PROP_FPS) or 30.0,
            max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            frames(),
        )
    finally:
        capture.release()


class VideoWriter:
    """
    Write PIL frames to an mp4; the size is taken from the first frame.

    Uses the ``mp4v`` codec, which OpenCV always has but many browsers will not
    play inline; re-encode with ffmpeg (H.264) if the video is for the web.
    Use as a context manager so the file is finalized even on an error.
    """

    def __init__(self, path: Path, fps: float) -> None:
        """Remember where to write; the file is created on the first frame."""
        self.path, self.fps = path, fps
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self) -> VideoWriter:
        """Return the writer itself."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Finish the file."""
        self.close()

    def write(self, frame: Image.Image) -> None:
        """Append one frame."""
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, frame.size
            )
        self._writer.write(
            cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        )

    def close(self) -> None:
        """Finish the file."""
        if self._writer is not None:
            self._writer.release()
