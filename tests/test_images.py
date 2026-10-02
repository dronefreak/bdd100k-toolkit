"""Image discovery for the demos."""

from __future__ import annotations

import pytest
from PIL import Image

from bdd100k_toolkit.utils.images import IMAGE_EXTENSIONS, list_images


def test_folder_listing_keeps_only_common_image_formats(tmp_path) -> None:  # noqa: ANN001
    assert len(IMAGE_EXTENSIONS) <= 15
    for name in ("b.PNG", "a.jpeg", "c.webp", "notes.txt", "d.json"):
        (tmp_path / name).touch()
    (tmp_path / "sub.jpg").mkdir()
    (tmp_path / "sub.jpg" / "x.jpg").touch()
    assert [p.name for p in list_images(tmp_path)] == ["a.jpeg", "b.PNG", "c.webp"]


def test_file_folder_and_bad_paths(tmp_path) -> None:  # noqa: ANN001
    image = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(image)
    assert list_images(image) == [image]
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No images"):
        list_images(empty)
