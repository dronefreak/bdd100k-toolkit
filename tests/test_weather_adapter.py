"""Tests for the BDD100K weather classification adapter."""

from __future__ import annotations

from pathlib import Path

from bdd100k_toolkit.classification.datasets.weather import BDD100KWeatherAdapter


def test_prepare_classification_writes_canonical_layout(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "canonical"
    BDD100KWeatherAdapter().prepare_classification(raw_bdd100k_dir, output_dir)

    # test split comes straight from val entries: 2 images, both valid weather.
    test_files = list((output_dir / "test").rglob("*.jpg"))
    assert len(test_files) == 2
    assert (output_dir / "test" / "foggy" / "v0.jpg").is_file()
    assert (output_dir / "test" / "clear" / "v1.jpg").is_file()

    # t4.jpg has weather="undefined": it is the `unknown` class by default.
    train_files = list((output_dir / "train").rglob("*.jpg"))
    valid_files = list((output_dir / "valid").rglob("*.jpg"))
    assert len(train_files) + len(valid_files) == 6
    unknown = list(output_dir.glob("*/unknown/t4.jpg"))
    assert len(unknown) == 1


def test_prepare_classification_can_exclude_unknown(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "canonical"
    BDD100KWeatherAdapter().prepare_classification(
        raw_bdd100k_dir, output_dir, include_unknown=False
    )
    files = [
        p for split in ("train", "valid") for p in (output_dir / split).rglob("*.jpg")
    ]
    assert len(files) == 5
    assert "t4.jpg" not in {p.name for p in files}
    assert not list(output_dir.glob("*/unknown"))


def test_prepare_classification_raises_on_missing_raw_dir(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        BDD100KWeatherAdapter().prepare_classification(
            tmp_path / "nope", tmp_path / "out"
        )
