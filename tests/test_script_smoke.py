"""
Smoke tests for the train/evaluate scripts: each runs end to end on tiny fixtures.

The heavy backends (Ultralytics, RF-DETR, segmentation-models-pytorch) are replaced
by small fakes, so these check the wiring (config, arguments, outputs) on CPU in
seconds, not the libraries themselves.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir

from bdd100k_toolkit.detection.trainer import YOLODetectionTrainer
from bdd100k_toolkit.scripts import (
    evaluate_semantic_seg,
    prepare_semantic_seg,
    train_classification,
    train_detection,
    train_detection_rfdetr,
    train_semantic_seg,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "src/bdd100k_toolkit/configs"


def _config(name: str, overrides: list[str]) -> Any:
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(name, overrides=overrides)


class _FakeSegModel(torch.nn.Module):
    """Stands in for an SMP model: a 1x1 conv from RGB to class logits."""

    def __init__(self, encoder_name: str, classes: int, activation: None) -> None:
        super().__init__()
        self.head = torch.nn.Conv2d(3, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # type: ignore[no-any-return]


@pytest.fixture
def fake_smp(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("segmentation_models_pytorch")
    module.Unet = _FakeSegModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", module)


@pytest.fixture
def seg_canonical(
    raw_bdd100k_semantic_seg_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Run ``bdd100k-prepare`` for segmentation and return the canonical root."""
    out = raw_bdd100k_semantic_seg_dir.parent / "seg"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bdd100k-prepare",
            "--dataset",
            "bdd100k-semantic-seg",
            "--raw-dir",
            str(raw_bdd100k_semantic_seg_dir),
            "--output-dir",
            str(out),
        ],
    )
    prepare_semantic_seg.main()
    return out


def test_semantic_seg_train_then_evaluate(
    fake_smp: None,
    seg_canonical: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    cfg = _config(
        "config_semantic_seg",
        [
            f"dataset.data_dir={seg_canonical}",
            "training.epochs=1",
            "training.batch_size=2",
            "training.imgsz=8",
            "training.workers=0",
            "training.device=cpu",
            f"training.output_dir={run}",
        ],
    )
    train_semantic_seg.main(cfg)
    checkpoint = run / "model.pt"
    assert checkpoint.is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bdd100k-evaluate",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            "bdd100k-semantic-seg",
            "--data-dir",
            str(seg_canonical),
            "--imgsz",
            "8",
            "--device",
            "cpu",
        ],
    )
    evaluate_semantic_seg.main()
    metrics = json.loads((run / "metrics.json").read_text())
    assert 0.0 <= metrics["miou"] <= 1.0
    assert len(metrics["per_class_iou"]) == 19


def test_train_classification_passes_the_config_to_the_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class FakeTrainer:
        def train(self, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {"model_path": str(tmp_path / "best.pt")}

    def fake_build(backend: str, name: str, device: str) -> FakeTrainer:
        seen.update(backend=backend, name=name, device=device)
        return FakeTrainer()

    monkeypatch.setattr(train_classification, "build_trainer", fake_build)
    cfg = _config(
        "config_classification",
        [
            f"dataset.data_dir={tmp_path}",
            "training.device=cpu",
            "training.epochs=3",
            "+training.extra.balance=true",
        ],
    )
    train_classification.main(cfg)
    assert seen["backend"] == "ultralytics"
    assert seen["epochs"] == 3
    assert seen["monitor"] == "macro_f1"
    assert seen["balance"] is True


def test_train_detection_checks_the_eval_split_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTrainer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def train(self, **kwargs: Any) -> dict[str, Any]:
            return {"model_path": str(tmp_path / "best.pt"), "output_dir": ""}

    metrics = {"mAP50": 0.5, "mAP50_95": 0.3, "per_class": {}}
    monkeypatch.setattr(train_detection, "YOLODetectionTrainer", FakeTrainer)
    monkeypatch.setattr(train_detection, "evaluate_detection", lambda _: metrics)

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\nnc: 10\n")
    overrides = [
        f"dataset.dataset_yaml={data_yaml}",
        "training.device=cpu",
        f"training.output_dir={tmp_path / 'run'}",
    ]
    with pytest.raises(ValueError, match="no 'test:' split"):
        train_detection.train_and_evaluate(_config("config_detection", overrides))

    cfg = _config("config_detection", [*overrides, "evaluation.split=valid"])
    train_detection.train_and_evaluate(cfg)
    saved = json.loads((tmp_path / "run" / "evaluation" / "metrics.json").read_text())
    assert saved["mAP50_95"] == 0.3


def test_train_detection_rfdetr_builds_the_model_and_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    class FakeRFDETR:
        def __init__(self, **kwargs: Any) -> None:
            calls["model"] = kwargs

        def train(self, **kwargs: Any) -> None:
            calls["train"] = kwargs

    monkeypatch.setattr(
        train_detection_rfdetr, "load_model_class", lambda _: FakeRFDETR
    )
    cfg = _config(
        "config_detection_rfdetr",
        [
            f"dataset.dataset_dir={tmp_path}",
            "training.device=cpu",
            "training.epochs=2",
            f"training.output_dir={tmp_path / 'run'}",
        ],
    )
    train_detection_rfdetr.train(cfg)
    assert calls["model"]["device"] == "cpu"
    assert calls["train"]["epochs"] == 2
    assert calls["train"]["dataset_dir"] == str(tmp_path.resolve())


def test_yolo_detection_trainer_maps_arguments_and_finds_best_pt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            seen["weights"] = weights

        def train(self, **kwargs: Any) -> str:
            seen.update(kwargs)
            weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"")
            return "results"

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)

    trainer = YOLODetectionTrainer("yolo11n", device="cpu")
    result = trainer.train(
        "data.yaml", optimizer="adamw", augment=False, output_dir=tmp_path, mosaic=0.5
    )
    assert seen["weights"] == "yolo11n.pt"
    assert seen["optimizer"] == "AdamW"
    assert seen["fliplr"] == 0.0  # augmentation off ...
    assert seen["mosaic"] == 0.5  # ... but explicit hyperparameters still win
    assert result["model_path"].endswith("yolo11n/weights/best.pt")
