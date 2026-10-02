"""Finding and loading the images a demo should run on."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

# The common formats, matched case-insensitively. Pillow opens most out of the
# box; ``.avif`` and ``.jp2`` need a Pillow build or plugin that supports them.
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".avif",
    ".jp2",
    ".ppm",
    ".pgm",
    ".tga",
)


def list_media(
    path: Path, extensions: tuple[str, ...] = IMAGE_EXTENSIONS, kind: str = "images"
) -> list[Path]:
    """Return ``[path]`` for a file, or the matching files directly in a folder."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"File or folder not found: {path}")
    found = sorted(
        (p for p in path.iterdir() if p.suffix.lower() in extensions and p.is_file()),
        key=lambda p: p.name.lower(),
    )
    if not found:
        raise ValueError(f"No {kind} ({' '.join(extensions)}) in {path}")
    return found


def list_images(path: Path) -> list[Path]:
    """Return ``[path]`` for an image file, or the images directly inside a folder."""
    return list_media(path)


def load_image(path: Path) -> Image.Image:
    """Open ``path`` as RGB, upright according to its EXIF orientation."""
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")
