"""Tests for the BDD100K scenario (scene) classification adapter."""

from __future__ import annotations

from pathlib import Path

from bdd100k_toolkit.classification.datasets.scenario import BDD100KScenarioAdapter


def test_prepare_classification_writes_canonical_layout(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "canonical"
    BDD100KScenarioAdapter().prepare_classification(raw_bdd100k_dir, output_dir)

    test_files = list((output_dir / "test").rglob("*.jpg"))
    assert len(test_files) == 2
    assert (output_dir / "test" / "gas stations" / "v0.jpg").is_file()
    assert (output_dir / "test" / "city street" / "v1.jpg").is_file()

    train_files = list((output_dir / "train").rglob("*.jpg"))
    valid_files = list((output_dir / "valid").rglob("*.jpg"))
    assert len(train_files) + len(valid_files) == 5
