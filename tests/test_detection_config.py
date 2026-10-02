"""The detection training config exposes everything the CLI forwards to Ultralytics."""

from __future__ import annotations

from pathlib import Path

import bdd100k_toolkit
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_detection_config_defaults_and_extra_override() -> None:
    configs = Path(bdd100k_toolkit.__file__).parent / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(configs)):
        cfg = compose(
            config_name="config_detection",
            overrides=["dataset.dataset_yaml=x.yaml", "+training.extra.fraction=0.1"],
        )
    training = cfg.training
    assert (training.optimizer, training.cos_lr, training.imgsz) == ("SGD", True, 960)
    extra = OmegaConf.to_container(training.extra, resolve=True)
    assert extra == {"warmup_epochs": 1.0, "close_mosaic": 10, "fraction": 0.1}
