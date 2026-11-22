"""Detection evaluation splits: ``valid`` -> Ultralytics ``val``, and early checks."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ultralytics")

from bdd100k_toolkit.scripts.evaluate_detection import (  # noqa: E402
    require_yaml_split,
    ultralytics_split,
)


@pytest.mark.parametrize(
    ("ours", "theirs"), [("valid", "val"), ("test", "test"), ("train", "train")]
)
def test_ultralytics_split(ours: str, theirs: str) -> None:
    assert ultralytics_split(ours) == theirs


def test_require_yaml_split_rejects_a_missing_split(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\nnc: 1\n")
    require_yaml_split(data_yaml, "valid")
    with pytest.raises(ValueError, match="no 'test:' split"):
        require_yaml_split(data_yaml, "test")
