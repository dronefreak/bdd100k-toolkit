"""
Classification training via the Ultralytics engine.

Ultralytics' classification trainer (``yolo11n-cls``, ``yolov8n-cls``, ...)
reads the same plain ImageFolder layout our adapters produce directly
(``data=<canonical_root>`` with ``train/valid/test`` subfolders of per-class
images), so, like DetectionBench's ``YOLOTrainer``, this class is a thin
wrapper: it delegates the actual training loop, augmentation, and loss to
Ultralytics rather than reimplementing any of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
            patience: Epochs with no improvement before early stopping.
            optimizer: Ultralytics optimizer name. ``"auto"`` lets Ultralytics
                choose the optimizer *and* its learning rate, ignoring ``lr``;
                pass e.g. ``"AdamW"`` for ``lr`` to take effect.
            **extra_kwargs: Passed directly to ``ultralytics.YOLO.train()``.

        Returns:
            dict with keys: 'results', 'model_path', 'output_dir'.

        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            model = self._UltralyticsYOLO(self._pt_name)
        except Exception as err:
            raise ValueError(
                f"Failed to initialize model '{self._pt_name}'. Browse the Ultralytics "
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
        }
