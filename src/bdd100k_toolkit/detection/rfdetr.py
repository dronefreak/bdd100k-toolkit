"""RF-DETR helpers shared by the train and evaluate scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

RFDETR_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}


def normalize_model_name(model_name: str) -> str:
    """Map aliases such as ``nano`` or ``rfdetr_nano`` to ``rfdetr-nano``."""
    name = model_name.strip().lower().replace("_", "-")
    return name if name.startswith("rfdetr") else f"rfdetr-{name}"


def is_rfdetr(model_name: str | None) -> bool:
    """Return True when ``model_name`` names an RF-DETR model."""
    return model_name is not None and normalize_model_name(model_name) in RFDETR_CLASSES


def load_model_class(model_name: str) -> type[Any]:
    """Return the ``rfdetr`` class for ``model_name`` (needs the ``rfdetr`` extra)."""
    canonical = normalize_model_name(model_name)
    if canonical not in RFDETR_CLASSES:
        supported = ", ".join(sorted(RFDETR_CLASSES))
        raise ValueError(f"Unsupported RF-DETR model '{model_name}': {supported}.")
    try:
        import rfdetr
    except ImportError as err:
        raise ImportError(
            'RF-DETR is not installed: pip install "bdd100k-toolkit[rfdetr]"'
        ) from err
    return getattr(rfdetr, RFDETR_CLASSES[canonical])  # type: ignore[no-any-return]


def read_training_resolution(checkpoint_path: str | Path | None) -> int | None:
    """
    Return the resolution a checkpoint was trained at, if recorded.

    RF-DETR checkpoints do not store it, but training writes
    ``training_config.json`` (``model_config.resolution``) beside them.
    """
    if not checkpoint_path:
        return None
    config_path = Path(checkpoint_path).parent / "training_config.json"
    try:
        resolution = json.loads(config_path.read_text())["model_config"]["resolution"]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return int(resolution) if resolution is not None else None


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


def build_model_kwargs(cfg: DictConfig, device: str) -> dict[str, Any]:
    """Build the RF-DETR model constructor kwargs from the Hydra config."""
    kwargs: dict[str, Any] = {"device": device}
    if cfg.model.num_classes is not None:
        kwargs["num_classes"] = int(cfg.model.num_classes)
    if cfg.model.pretrain_weights:
        kwargs["pretrain_weights"] = str(Path(cfg.model.pretrain_weights).resolve())
    if cfg.model.resolution is not None:
        kwargs["resolution"] = int(cfg.model.resolution)
    return kwargs


def build_training_kwargs(cfg: DictConfig, device: str) -> dict[str, Any]:
    """Build the RF-DETR ``train()`` kwargs from the Hydra config."""
    t = cfg.training
    batch_size = "auto" if str(t.batch_size).lower() == "auto" else int(t.batch_size)
    kwargs: dict[str, Any] = {
        "dataset_dir": str(Path(t.dataset_dir).resolve()),
        "dataset_file": str(t.dataset_file),
        "output_dir": str(Path(t.output_dir).resolve()),
        "epochs": int(t.epochs),
        "batch_size": batch_size,
        "grad_accum_steps": int(t.grad_accum_steps),
        "lr": float(t.learning_rate),
        "lr_encoder": float(t.learning_rate_encoder),
        "lr_scheduler": str(t.scheduler_type),
        "warmup_epochs": float(t.warmup_epochs),
        "checkpoint_interval": int(t.checkpoint_interval),
        "eval_interval": int(t.eval_interval),
        "num_workers": int(t.workers),
        "weight_decay": float(t.weight_decay),
        "use_ema": bool(t.use_ema),
        "tensorboard": bool(t.tensorboard),
        "wandb": bool(t.wandb),
        "early_stopping": bool(t.early_stopping),
        "early_stopping_patience": int(t.early_stopping_patience),
        "early_stopping_min_delta": float(t.early_stopping_min_delta),
        "progress_bar": t.progress_bar,
        "amp_dtype": str(t.amp_dtype),
        "device": device,
        "run_test": bool(t.run_test),
        "augmentation_backend": str(t.augmentation_backend),
        "seed": int(t.seed),
    }
    if cfg.model.resolution is not None:
        kwargs["resolution"] = int(cfg.model.resolution)
    if t.resume:
        kwargs["resume"] = str(t.resume)
    return kwargs
