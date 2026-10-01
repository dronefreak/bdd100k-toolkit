"""
Classification training via the Ultralytics engine.

Ultralytics' classification trainer (``yolo11n-cls``, ``yolov8n-cls``, ...)
reads the same plain ImageFolder layout our adapters produce directly
(``data=<canonical_root>`` with ``train/valid/test`` subfolders of per-class
images), so, like DetectionBench's ``YOLOTrainer``, this class is a thin
wrapper: it delegates the actual training loop, augmentation, and loss to
Ultralytics rather than reimplementing any of it.

What Ultralytics provides, and what this wrapper changes:

- **EMA**: always on in Ultralytics (``ModelEMA``); ``best.pt`` and ``last.pt``
  hold the EMA weights. There is no switch to turn it off.
- **Early stopping**: ``patience`` (epochs without improvement of the monitored
  metric).
- **Best-checkpoint saving**: by the monitored metric. Ultralytics' own
  classification fitness is ``(top-1 + top-5) / 2``, which with at most 7
  classes is almost always just top-1. This wrapper replaces it with the
  ``monitor`` metric (macro F1 by default), so ``best.pt``, early stopping and
  ``best_fitness`` all follow a metric that is meaningful on imbalanced data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bdd100k_toolkit.utils.checks import require_split_dirs
from bdd100k_toolkit.utils.cls_metrics import (
    DEFAULT_MONITOR,
    check_monitor,
    monitor_value,
)


def monitor_from_predictions(
    targets: list[Any], predictions: list[Any], num_classes: int, monitor: str
) -> float:
    """
    Compute ``monitor`` from Ultralytics' accumulated validation tensors.

    ``targets`` is a list of ``(n,)`` class-id tensors and ``predictions`` a list
    of ``(n, k)`` tensors of the top-k classes (best first); only the top-1
    column is used.
    """
    true = np.concatenate([np.asarray(t).reshape(-1) for t in targets])
    predicted = np.concatenate([np.asarray(p)[:, 0] for p in predictions])
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (true.astype(int), predicted.astype(int)), 1)
    return monitor_value(confusion, monitor)


def make_trainer_class(monitor: str) -> type:
    """
    Build a ``ClassificationTrainer`` subclass whose fitness is ``monitor``.

    Ultralytics uses one number, the validation "fitness", for best-checkpoint
    selection, early stopping and ``best_fitness``. The returned class swaps the
    validator's fitness for the ``monitor`` metric and leaves everything else
    (EMA, schedule, augmentation, saving) untouched.
    """
    from ultralytics.models.yolo.classify import ClassificationTrainer

    check_monitor(monitor)

    class MonitorClassificationTrainer(ClassificationTrainer):  # type: ignore[misc]
        """ClassificationTrainer that selects checkpoints by ``monitor``."""

        def get_validator(self) -> Any:
            validator = super().get_validator()
            original_get_stats = validator.get_stats

            def get_stats() -> dict[str, float]:
                stats = original_get_stats()  # top1, top5 and the default fitness
                stats["fitness"] = monitor_from_predictions(
                    validator.targets,
                    validator.pred,
                    len(validator.names),
                    monitor,
                )
                return stats

            validator.get_stats = get_stats
            return validator

    return MonitorClassificationTrainer


class UltralyticsClassificationTrainer:
    """Trains image classifiers using the Ultralytics training engine."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        """
        Initialize UltralyticsClassificationTrainer.

        Args:
            model_name: Registered Ultralytics classification checkpoint
                name, e.g. 'yolo11n-cls', 'yolov8s-cls'.
            device: Device string passed to Ultralytics ('cuda', 'cpu', ...).

        """
        try:
            from ultralytics import YOLO as UltralyticsYOLO
        except ImportError as err:
            raise ImportError(
                "Ultralytics is required for classification training. "
                "Install with: pip install ultralytics>=8.0.0"
            ) from err

        self._pt_name = f"{model_name}.pt"
        self._model_name = model_name
        self.device = device
        self._UltralyticsYOLO = UltralyticsYOLO

    def train(  # noqa: PLR0913, PLR0917
        self,
        data_dir: str | Path,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float | None = None,
        imgsz: int = 224,
        output_dir: str | Path = "outputs",
        workers: int = 4,
        patience: int = 100,
        optimizer: str = "auto",
        monitor: str = DEFAULT_MONITOR,
        pretrained: bool = True,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train a classifier on the canonical ``{train,valid,test}/<class>/`` layout.

        Args:
            data_dir: Canonical classification root produced by an adapter's
                ``prepare_classification`` (must contain a ``train/`` and,
                ideally, a ``valid/`` subfolder; Ultralytics discovers both
                by directory name automatically).
            epochs: Number of training epochs.
            batch_size: Batch size.
            lr: Initial learning rate (``lr0`` in Ultralytics terminology);
                None means 0.001.
            imgsz: Input image size (classification defaults to 224, unlike
                detection's 640; most classification backbones are pretrained
                at that resolution).
            output_dir: Where to save the final model and logs.
            workers: Number of DataLoader workers.
            patience: Epochs with no improvement of ``monitor`` before early
                stopping (0 disables it).
            optimizer: Ultralytics optimizer name. ``"auto"`` lets Ultralytics
                choose the optimizer *and* its learning rate, ignoring ``lr``;
                pass e.g. ``"AdamW"`` for ``lr`` to take effect.
            monitor: Metric that selects ``best.pt`` and drives early stopping:
                ``"macro_f1"`` (default), ``"balanced_accuracy"`` or
                ``"accuracy"``, computed on ``valid`` every epoch.
            pretrained: Start from the pretrained ``.pt`` weights; False builds
                the architecture from its ``.yaml`` and trains from scratch.
            **extra_kwargs: Passed directly to ``ultralytics.YOLO.train()``.

        Returns:
            dict with keys: 'results', 'model_path', 'output_dir'.

        """
        check_monitor(monitor)
        require_split_dirs(Path(data_dir), ("train",))
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        source = self._pt_name if pretrained else f"{self._model_name}.yaml"
        try:
            model = self._UltralyticsYOLO(source)
        except Exception as err:
            raise ValueError(
                f"Failed to initialize model '{source}'. Browse the Ultralytics "
                "classification models at https://docs.ultralytics.com/tasks/classify: "
                f"{err}"
            ) from err
        train_kwargs: dict[str, Any] = {
            "data": str(Path(data_dir).resolve()),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "lr0": 0.001 if lr is None else lr,
            "device": self.device,
            "workers": workers,
            "patience": patience,
            "optimizer": optimizer,
            "project": str(output_dir),
            "name": self._model_name,
            "exist_ok": True,
            "trainer": make_trainer_class(monitor),
        }
        train_kwargs.update(extra_kwargs)

        results = model.train(**train_kwargs)

        weights_dir = output_dir / self._model_name / "weights"
        best_model = weights_dir / "best.pt"
        last_model = weights_dir / "last.pt"
        final_path = best_model if best_model.exists() else last_model

        return {
            "results": results,
            "model_path": str(final_path) if final_path.exists() else None,
            "output_dir": str(output_dir / self._model_name),
            "monitor": monitor,
        }
