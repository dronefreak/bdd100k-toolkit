"""The Ultralytics backend with the real library (tiny, CPU, offline, from scratch)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import torch  # noqa: E402

from bdd100k_toolkit.classification.ultralytics_trainer import (  # noqa: E402
    UltralyticsClassificationTrainer,
)


def _train(data: Path, out: Path, **kwargs: object) -> dict:
    params: dict = {
        "epochs": 2,
        "batch_size": 6,
        "imgsz": 32,
        "workers": 0,
        "pretrained": False,  # build from yaml: no weight download
        "output_dir": out,
        "plots": False,
        "amp": False,
        "verbose": False,
    }
    params.update(kwargs)
    return UltralyticsClassificationTrainer("yolo11n-cls", device="cpu").train(
        data, **params
    )


def test_fitness_is_the_monitored_metric_not_ultralytics_default(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    result = _train(tiny_dataset, tmp_path / "o", monitor="accuracy")
    checkpoint = torch.load(result["model_path"], weights_only=False)
    metrics = checkpoint["train_metrics"]
    # Ultralytics' default would be (top1 + top5) / 2; with 3 classes top5 is 1.0
    assert metrics["fitness"] == pytest.approx(
        metrics["metrics/accuracy_top1"], abs=1e-4
    )


def test_macro_f1_monitor_runs_end_to_end(tiny_dataset: Path, tmp_path: Path) -> None:
    result = _train(tiny_dataset, tmp_path / "o", monitor="macro_f1")
    metrics = torch.load(result["model_path"], weights_only=False)["train_metrics"]
    assert 0.0 <= metrics["fitness"] <= 1.0
    assert result["monitor"] == "macro_f1"


def test_patience_stops_training_early(tiny_dataset: Path, tmp_path: Path) -> None:
    result = _train(
        tiny_dataset, tmp_path / "o", epochs=20, lr=1e-9, optimizer="SGD", patience=1
    )
    csv_path = Path(result["output_dir"]) / "results.csv"
    epochs_run = len(csv_path.read_text().strip().splitlines()) - 1
    assert epochs_run < 20
