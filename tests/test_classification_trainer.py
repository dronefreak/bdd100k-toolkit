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


_SENTINEL_TRAINER = type("SentinelTrainer", (), {})


@pytest.fixture(autouse=True)
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeYOLO.calls = []
    # make_trainer_class imports the real Ultralytics, which the fake replaces
    monkeypatch.setattr(
        "bdd100k_toolkit.classification.ultralytics_trainer.make_trainer_class",
        lambda monitor: _SENTINEL_TRAINER,
    )
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


def test_monitor_is_wired_into_a_custom_trainer(tmp_path: Path) -> None:
    trainer = UltralyticsClassificationTrainer("yolo11n-cls", device="cpu")
    result = trainer.train(tmp_path, output_dir=tmp_path / "o", monitor="accuracy")
    assert _FakeYOLO.calls[0]["trainer"] is _SENTINEL_TRAINER
    assert result["monitor"] == "accuracy"


def test_unknown_monitor_is_rejected(tmp_path: Path) -> None:
    trainer = UltralyticsClassificationTrainer("yolo11n-cls", device="cpu")
    with pytest.raises(ValueError, match="monitor"):
        trainer.train(tmp_path, output_dir=tmp_path / "o", monitor="top5")


def test_pretrained_false_builds_from_yaml(tmp_path: Path) -> None:
    trainer = UltralyticsClassificationTrainer("yolo11n-cls", device="cpu")
    seen: list[str] = []
    original = _FakeYOLO.__init__

    def spy(self: _FakeYOLO, name: str) -> None:
        seen.append(name)
        original(self, name)

    _FakeYOLO.__init__ = spy  # type: ignore[method-assign]
    try:
        trainer.train(tmp_path, output_dir=tmp_path / "o", pretrained=False)
        trainer.train(tmp_path, output_dir=tmp_path / "o", pretrained=True)
    finally:
        _FakeYOLO.__init__ = original  # type: ignore[method-assign]
    assert seen == ["yolo11n-cls.yaml", "yolo11n-cls.pt"]


def test_monitor_from_predictions_uses_the_top1_column() -> None:
    import numpy as np

    from bdd100k_toolkit.classification.ultralytics_trainer import (
        monitor_from_predictions,
    )

    targets = [np.array([0, 0, 1, 1])]
    # top-1 is the first column; the second column (top-2) must be ignored
    predictions = [np.array([[0, 1], [1, 0], [1, 0], [1, 0]])]
    assert monitor_from_predictions(targets, predictions, 2, "accuracy") == 0.75
    f1 = monitor_from_predictions(targets, predictions, 2, "macro_f1")
    assert f1 == pytest.approx((2 / 3 + 0.8) / 2)
