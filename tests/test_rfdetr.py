from __future__ import annotations

import json
from pathlib import Path

import bdd100k_toolkit
import pytest
from bdd100k_toolkit.detection.rfdetr import (
    build_model_kwargs,
    build_training_kwargs,
    is_rfdetr,
    normalize_model_name,
    read_training_resolution,
)
from hydra import compose, initialize_config_dir


def _cfg(*overrides: str):
    configs = Path(bdd100k_toolkit.__file__).parent / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(configs)):
        return compose(
            config_name="config_detection_rfdetr",
            overrides=["dataset.dataset_dir=/data/coco", *overrides],
        )


def test_model_name_aliases() -> None:
    assert normalize_model_name("Nano") == "rfdetr-nano"
    assert normalize_model_name("rfdetr_small") == "rfdetr-small"
    assert is_rfdetr("rfdetr-medium")
    assert not is_rfdetr("yolo11n")
    assert not is_rfdetr(None)


def test_kwargs_follow_config() -> None:
    cfg = _cfg("model.resolution=640", "training.batch_size=8")
    model_kwargs = build_model_kwargs(cfg, "cuda")
    assert model_kwargs == {"device": "cuda", "num_classes": 10, "resolution": 640}
    kwargs = build_training_kwargs(cfg, "cuda")
    assert kwargs["batch_size"] == 8
    assert kwargs["dataset_dir"] == "/data/coco"
    assert kwargs["resolution"] == 640
    assert "resume" not in kwargs
    assert build_training_kwargs(_cfg(), "cuda")["batch_size"] == "auto"


def test_read_training_resolution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_best_total.pth"
    assert read_training_resolution(checkpoint) is None
    (tmp_path / "training_config.json").write_text(
        json.dumps({"model_config": {"resolution": 576}})
    )
    assert read_training_resolution(checkpoint) == 576


def test_unknown_model_is_rejected() -> None:
    from bdd100k_toolkit.detection.rfdetr import load_model_class

    with pytest.raises(ValueError, match="Unsupported RF-DETR model"):
        load_model_class("rfdetr-huge")
