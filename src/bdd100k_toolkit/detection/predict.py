"""
Single-image inference for Ultralytics detection checkpoints.

``Detector(checkpoint)`` loads a detection ``best.pt`` and maps a PIL image to
:class:`Detections`: boxes in pixel coordinates, scores, class ids and the
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
    """A loaded detection checkpoint."""

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
            checkpoint: A detection ``best.pt`` / ``last.pt`` written by Ultralytics.
            device: ``"auto"`` (CUDA if available), ``"cpu"``, ``"cuda"`` ...
            conf: Minimum score of a returned box.
            iou: NMS IoU threshold.
            imgsz: Inference size; defaults to the size the model was trained at.

        """
        from ultralytics import YOLO

        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        self._yolo = YOLO(str(checkpoint))
        if self._yolo.task != "detect":
            raise ValueError(
                f"{checkpoint} is a '{self._yolo.task}' model, not a detector"
            )
        self._device = resolve_device(device)
        self._options: dict[str, Any] = {"conf": conf, "iou": iou}
        if imgsz:
            self._options["imgsz"] = imgsz
        names = self._yolo.names
        self.class_names: list[str] = [names[i] for i in sorted(names)]
        self.model_name = Path(checkpoint).parent.parent.name

    def predict(self, image: Image.Image) -> Detections:
        """Detect objects in one image (any mode; converted to RGB)."""
        started = time.perf_counter()
        results: Any = self._yolo.predict(
            source=image.convert("RGB"),
            device=self._device,
            verbose=False,
            **self._options,
        )
        boxes = results[0].boxes
        return Detections(
            [tuple(map(float, b)) for b in boxes.xyxy.cpu().tolist()],  # type: ignore[misc]
            [float(s) for s in boxes.conf.cpu().tolist()],
            [int(c) for c in boxes.cls.cpu().tolist()],
            self.class_names,
            time.perf_counter() - started,
        )
