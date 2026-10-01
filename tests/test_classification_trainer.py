"""ClassificationTrainer wiring, with a fake Ultralytics (no GPU, no download)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

from bdd100k_toolkit.classification.ultralytics_trainer import (
    UltralyticsClassificationTrainer,
)
import pytest


class _FakeYOLO:
    calls: list[dict[str, Any]] = []  # noqa: RUF012

    def __init__(self, name: str) -> None:
        if name.startswith("bad"):
            raise FileNotFoundError(f"no such model: {name}")
        self.name = name

    def train(self, **kwargs: Any) -> str:
        _FakeYOLO.calls.append(kwargs)
        return "results"


@pytest.fixture(autouse=True)
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeYOLO.calls = []
    module = types.ModuleType("ultralytics")
    module.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def test_train_passes_core_arguments_and_extra_kwargs(tmp_path: Path) -> None:
    trainer = UltralyticsClassificationTrainer("yolo11n-cls", device="cpu")
    result = trainer.train(
        data_dir=tmp_path,
        epochs=3,
        batch_size=8,
        optimizer="AdamW",
        output_dir=tmp_path / "out",
        fraction=0.1,
    )
    kwargs = _FakeYOLO.calls[0]
    assert kwargs["data"] == str(tmp_path.resolve())
    assert kwargs["epochs"] == 3
    assert kwargs["batch"] == 8
    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["fraction"] == 0.1
    assert kwargs["device"] == "cpu"
    assert result["results"] == "results"
    assert result["model_path"] is None  # the fake wrote no weights


def test_optimizer_defaults_to_auto(tmp_path: Path) -> None:
    from bdd100k_toolkit.classification.ultralytics_trainer import (
        UltralyticsClassificationTrainer,
    )

    UltralyticsClassificationTrainer("yolo11n-cls").train(
        tmp_path, output_dir=tmp_path / "o"
    )
    assert _FakeYOLO.calls[0]["optimizer"] == "auto"


def test_unknown_model_gives_a_helpful_error_with_the_cause(tmp_path: Path) -> None:
    trainer = UltralyticsClassificationTrainer("bad-model", device="cpu")
    with pytest.raises(ValueError, match="bad-model.pt") as info:
        trainer.train(tmp_path, output_dir=tmp_path / "o")
    assert "docs.ultralytics.com" in str(info.value)
    assert isinstance(info.value.__cause__, FileNotFoundError)  # original kept
