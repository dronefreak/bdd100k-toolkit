"""Shrinking images during prepare: --max-width and friends."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bdd100k_toolkit import cli
from bdd100k_toolkit.classification import get
from bdd100k_toolkit.utils.io import ImageOptions, materialize_images, resize_image

WIDE = (640, 360)


def _save(
    path: Path, size: tuple[int, int], color: tuple[int, int, int] = (90, 60, 30)
) -> None:
    Image.new("RGB", size, color).save(path, quality=95)


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


@pytest.fixture
def real_raw_dir(raw_bdd100k_dir: Path) -> Path:
    """Return the fake official download, with real 640x360 JPEGs, not dummy bytes."""
    for jpg in (raw_bdd100k_dir / "images" / "100k").rglob("*.jpg"):
        _save(jpg, WIDE)
    return raw_bdd100k_dir


def test_resize_keeps_the_aspect_ratio_and_leaves_no_partial_files(
    tmp_path: Path,
) -> None:
    src, dst = tmp_path / "a.jpg", tmp_path / "out.jpg"
    _save(src, (1280, 720))
    assert resize_image(src, dst, 512, 90) == "resized"
    assert _size(dst) == (512, 288)
    assert not list(tmp_path.glob("*.part"))
    assert resize_image(src, dst, 512, 90) == "skipped"  # never overwrites


def test_odd_sizes_round_the_height(tmp_path: Path) -> None:
    src, dst = tmp_path / "a.jpg", tmp_path / "out.jpg"
    _save(src, (1001, 333))
    resize_image(src, dst, 500, 90)
    assert _size(dst) == (500, round(333 * 500 / 1001))


def test_images_already_narrow_enough_are_hardlinked_never_upscaled(
    tmp_path: Path,
) -> None:
    src, dst = tmp_path / "a.jpg", tmp_path / "out.jpg"
    _save(src, (400, 225))
    assert resize_image(src, dst, 512, 90) == "linked"
    assert _size(dst) == (400, 225)
    assert os.path.samefile(src, dst)


def test_a_corrupt_image_names_the_file(tmp_path: Path) -> None:
    src = tmp_path / "bad.jpg"
    src.write_bytes(b"not a jpeg")
    with pytest.raises(RuntimeError, match="bad.jpg"):
        resize_image(src, tmp_path / "out.jpg", 512, 90)


def test_option_validation() -> None:
    with pytest.raises(ValueError, match="max_width"):
        ImageOptions(max_width=10)
    with pytest.raises(ValueError, match="jpeg_quality"):
        ImageOptions(jpeg_quality=0)
    with pytest.raises(ValueError, match="jpeg_quality"):
        ImageOptions(jpeg_quality=101)
    with pytest.raises(ValueError, match="workers"):
        ImageOptions(workers=0)
    assert not ImageOptions().shrinks("train")  # off by default
    options = ImageOptions(max_width=256)
    assert options.shrinks("train") and options.shrinks("valid")
    assert not options.shrinks("test")  # reported numbers stay on the originals
    assert ImageOptions(max_width=256, shrink_test=True).shrinks("test")


def test_parallel_and_serial_resizing_give_identical_pixels(tmp_path: Path) -> None:
    jobs = []
    for i in range(6):
        src = tmp_path / f"s{i}.jpg"
        _save(src, (1280, 720), (i * 30, 100, 200 - i * 20))
        jobs.append((src, tmp_path / "serial" / f"{i}.jpg"))
    (tmp_path / "serial").mkdir()
    parallel_jobs = [(s, tmp_path / "parallel" / d.name) for s, d in jobs]
    (tmp_path / "parallel").mkdir()
    serial = materialize_images(
        jobs, ImageOptions(max_width=256, workers=1), shrink=True
    )
    parallel = materialize_images(
        parallel_jobs, ImageOptions(max_width=256, workers=2), shrink=True
    )
    assert serial["resized"] == parallel["resized"] == 6
    for (_, a), (_, b) in zip(jobs, parallel_jobs, strict=True):
        assert np.array_equal(np.asarray(Image.open(a)), np.asarray(Image.open(b)))


def test_prepare_shrinks_train_and_valid_but_not_test_by_default(
    real_raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    get("bdd100k-weather").prepare_classification(
        real_raw_dir, out, image_options=ImageOptions(max_width=320, workers=1)
    )
    for split in ("train", "valid"):
        for jpg in (out / split).rglob("*.jpg"):
            assert _size(jpg) == (320, 180), jpg
    for jpg in (out / "test").rglob("*.jpg"):
        assert _size(jpg) == WIDE
        source = real_raw_dir / "images" / "100k" / "val" / jpg.name
        assert os.path.samefile(jpg, source)  # still a hardlink to the original


def test_shrink_test_also_shrinks_the_test_split(
    real_raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    get("bdd100k-weather").prepare_classification(
        real_raw_dir,
        out,
        image_options=ImageOptions(max_width=320, shrink_test=True, workers=1),
    )
    assert {_size(p) for p in (out / "test").rglob("*.jpg")} == {(320, 180)}


def test_shrinking_does_not_change_which_images_land_where(
    real_raw_dir: Path, tmp_path: Path
) -> None:
    plain, small = tmp_path / "plain", tmp_path / "small"
    adapter = get("bdd100k-weather")
    adapter.prepare_classification(real_raw_dir, plain)
    adapter.prepare_classification(
        real_raw_dir, small, image_options=ImageOptions(max_width=320, workers=1)
    )

    def listing(root: Path) -> set[tuple[str, str, str]]:
        return {(p.parts[-3], p.parts[-2], p.name) for p in root.glob("*/*/*.jpg")}

    assert listing(plain) == listing(small)


def test_prepare_info_records_how_the_dataset_was_made(
    real_raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    get("bdd100k-weather").prepare_classification(
        real_raw_dir,
        out,
        include_unknown=False,
        image_options=ImageOptions(max_width=320, jpeg_quality=80, workers=1),
    )
    info = json.loads((out / "prepare_info.json").read_text())
    assert info["attribute"] == "weather"
    assert info["max_width"] == 320 and info["jpeg_quality"] == 80
    assert info["shrink_test"] is False and info["include_unknown"] is False
    assert info["layout"] == "official"
    assert sum(info["images_per_split"].values()) == 7  # 5 train/valid + 2 test
    # plain runs record that nothing was shrunk
    plain = tmp_path / "plain"
    get("bdd100k-weather").prepare_classification(real_raw_dir, plain)
    plain_info = json.loads((plain / "prepare_info.json").read_text())
    assert plain_info["max_width"] is None and plain_info["jpeg_quality"] is None


def test_a_non_empty_output_dir_is_warned_about(
    real_raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    get("bdd100k-weather").prepare_classification(real_raw_dir, out)
    with pytest.warns(UserWarning, match="not empty"):
        get("bdd100k-weather").prepare_classification(
            real_raw_dir, out, image_options=ImageOptions(max_width=320, workers=1)
        )


def test_resizing_works_from_the_kaggle_folder_layout_too(tmp_path: Path) -> None:
    for split, count in {"train": 6, "val": 2}.items():
        for name in ("clear", "unknown"):
            folder = tmp_path / "kaggle" / split / name
            folder.mkdir(parents=True)
            for i in range(count):
                _save(folder / f"{name}{i}.jpg", WIDE)
    out = tmp_path / "out"
    get("bdd100k-weather").prepare_classification(
        tmp_path / "kaggle", out, image_options=ImageOptions(max_width=320, workers=1)
    )
    assert {_size(p) for p in (out / "train").rglob("*.jpg")} == {(320, 180)}
    assert {_size(p) for p in (out / "test").rglob("*.jpg")} == {WIDE}


def test_cli_flags_reach_the_adapter(
    monkeypatch: pytest.MonkeyPatch, real_raw_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bdd100k-prepare",
            "--dataset",
            "bdd100k-weather",
            "--raw-dir",
            str(real_raw_dir),
            "--output-dir",
            str(out),
            "--max-width",
            "320",
            "--workers",
            "1",
        ],
    )
    cli.prepare()
    assert {_size(p) for p in (out / "train").rglob("*.jpg")} == {(320, 180)}
