"""Classification training backends, selected by ``model.backend`` in the config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

BACKENDS = ("ultralytics", "timm")
DEFAULT_BACKEND = "ultralytics"


class ClassificationBackend(Protocol):
    """What the train CLI needs from a backend."""

    def train(  # noqa: D102, PLR0913, PLR0917
        self,
        data_dir: str | Path,
        epochs: int,
        batch_size: int,
        lr: float | None,
        imgsz: int,
        output_dir: str | Path,
        workers: int,
        patience: int,
        optimizer: str,
        **extra: Any,
    ) -> dict[str, Any]: ...


def build_trainer(backend: str, model_name: str, device: str) -> ClassificationBackend:
    """Instantiate the trainer for ``backend`` (imports it lazily)."""
    if backend == "ultralytics":
        from bdd100k_toolkit.classification.ultralytics_trainer import (
            UltralyticsClassificationTrainer,
        )

        return UltralyticsClassificationTrainer(model_name, device=device)  # type: ignore[return-value]
    if backend == "timm":
        from bdd100k_toolkit.classification.timm_trainer import (
            TimmClassificationTrainer,
        )

        return TimmClassificationTrainer(model_name, device=device)  # type: ignore[return-value]
    raise ValueError(f"Unknown backend '{backend}'. Supported: {', '.join(BACKENDS)}.")
