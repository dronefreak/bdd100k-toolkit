"""
Single-image inference for detection checkpoints.

``Detector(checkpoint)`` loads an Ultralytics ``best.pt`` or an RF-DETR
``checkpoint_best_total.pth`` (told apart by the file suffix) and maps a PIL image
to :class:`Detections`: boxes in pixel coordinates, scores, class ids and the
inference time. Nothing here draws or prints.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from bdd100k_toolkit.utils.device import resolve_device


@dataclass(frozen=True)
class Detections:
    """Boxes of one image: ``boxes[i]`` is ``(x1, y1, x2, y2)`` for ``labels[i]``."""

    boxes: list[tuple[float, float, float, float]]
    scores: list[float]
    labels: list[int]
    class_names: list[str]
    seconds: float

    def counts(self) -> Counter[str]:
        """Return how many boxes each class has."""
        return Counter(self.class_names[label] for label in self.labels)


class Detector:
    """A loaded detection checkpoint (Ultralytics ``.pt`` or RF-DETR ``.pth``)."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int | None = None,
    ) -> None:
        """
        Load ``checkpoint``.

        Args:
            checkpoint: An Ultralytics ``best.pt`` / ``last.pt``, or an RF-DETR
                ``.pth`` with its ``training_config.json`` beside it.
            device: ``"auto"`` (CUDA if available), ``"cpu"``, ``"cuda"`` ...
            conf: Minimum score of a returned box.
            iou: NMS IoU threshold (Ultralytics only; RF-DETR does not use NMS).
            imgsz: Inference size; defaults to the size the model was trained at.

        """
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        self._device = resolve_device(device)
        self._conf = conf
        self.model_name = Path(checkpoint).parent.name
        self.is_rfdetr = Path(checkpoint).suffix == ".pth"
        self.class_names: list[str]
        if self.is_rfdetr:
            self._load_rfdetr(checkpoint, imgsz)
        else:
            self._load_ultralytics(checkpoint, iou, imgsz)

    def _load_ultralytics(
        self, checkpoint: str | Path, iou: float, imgsz: int | None
    ) -> None:
        from ultralytics import YOLO

        self._yolo = YOLO(str(checkpoint))
        if self._yolo.task != "detect":
            raise ValueError(
                f"{checkpoint} is a '{self._yolo.task}' model, not a detector"
            )
        self._options: dict[str, Any] = {"conf": self._conf, "iou": iou}
        if imgsz:
            self._options["imgsz"] = imgsz
        names = self._yolo.names
        self.class_names = [names[i] for i in sorted(names)]
        self.model_name = Path(checkpoint).parent.parent.name

    def _load_rfdetr(self, checkpoint: str | Path, imgsz: int | None) -> None:
        from bdd100k_toolkit.detection.rfdetr import (
            load_model_class,
            read_training_config,
        )

        config = read_training_config(checkpoint)
        if config is None:
            raise ValueError(
                f"{checkpoint}: RF-DETR needs the training_config.json that training "
                "writes beside the checkpoint (model family, classes, resolution)."
            )
        model_config = config["model_config"]
        self.class_names = list(config["class_names"])
        kwargs: dict[str, Any] = {
            "device": self._device,
            "pretrain_weights": str(Path(checkpoint).resolve()),
            "num_classes": int(config["num_classes"]),
            "resolution": imgsz or int(model_config["resolution"]),
        }
        self._rfdetr = load_model_class(model_config["model_name"])(**kwargs)

    def predict(self, image: Image.Image) -> Detections:
        """Detect objects in one image (any mode; converted to RGB)."""
        started = time.perf_counter()
        image = image.convert("RGB")
        if self.is_rfdetr:
            found = self._rfdetr.predict(
                image, threshold=self._conf, include_source_image=False
            )
            boxes = [tuple(map(float, b)) for b in found.xyxy.tolist()]
            scores = [float(s) for s in found.confidence.tolist()]
            labels = [int(c) for c in found.class_id.tolist()]
        else:
            results: Any = self._yolo.predict(
                source=image, device=self._device, verbose=False, **self._options
            )
            raw = results[0].boxes
            boxes = [tuple(map(float, b)) for b in raw.xyxy.cpu().tolist()]
            scores = [float(s) for s in raw.conf.cpu().tolist()]
            labels = [int(c) for c in raw.cls.cpu().tolist()]
        return Detections(
            boxes,  # type: ignore[arg-type]
            scores,
            labels,
            self.class_names,
            time.perf_counter() - started,
        )
