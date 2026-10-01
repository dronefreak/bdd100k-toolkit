"""
Single-image inference for checkpoints from either classification backend.

``Predictor(checkpoint)`` loads a ``best.pt`` written by the ``ultralytics`` or the
``timm`` backend (detected from the file unless given) and maps a PIL image to a
:class:`Prediction`: one probability per class, plus the forward-pass time. It uses
the same preprocessing as evaluation, so a prediction matches what
``bdd100k-evaluate`` would say for that image. Nothing here draws or prints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from bdd100k_toolkit.utils.device import resolve_device

BACKENDS = ("auto", "ultralytics", "timm")


@dataclass(frozen=True)
class Prediction:
    """Class probabilities for one image; ``probabilities[i]`` is ``class_names[i]``."""

    class_names: list[str]
    probabilities: list[float]
    seconds: float

    def ranked(self) -> list[tuple[str, float]]:
        """Return ``(class, probability)`` pairs, most likely first."""
        pairs = zip(self.class_names, self.probabilities, strict=True)
        return sorted(pairs, key=lambda pair: -pair[1])

    @property
    def top1(self) -> tuple[str, float]:
        """The most likely class and its probability."""
        return self.ranked()[0]

    @property
    def margin(self) -> float:
        """Probability gap between the best and the second-best class."""
        ranked = self.ranked()
        return ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]


class Predictor:
    """A loaded classification checkpoint."""

    def __init__(
        self, checkpoint: str | Path, backend: str = "auto", device: str = "auto"
    ) -> None:
        """
        Load ``checkpoint``.

        Args:
            checkpoint: A ``best.pt`` / ``last.pt`` written by either backend.
            backend: ``"auto"`` (detect from the file), ``"ultralytics"`` or ``"timm"``.
            device: ``"auto"`` (CUDA if available), ``"cpu"``, ``"cuda"`` ...

        """
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if backend == "auto":
            from bdd100k_toolkit.classification.timm_trainer import is_timm_checkpoint

            backend = "timm" if is_timm_checkpoint(checkpoint) else "ultralytics"
        self.backend = backend
        name = resolve_device(device)
        self._device = torch.device(f"cuda:{name}" if name.isdigit() else name)
        self.model_name: str
        self.class_names: list[str]
        if backend == "timm":
            self._load_timm(checkpoint)
        else:
            self._load_ultralytics(checkpoint)

    def _load_timm(self, checkpoint: str | Path) -> None:
        from bdd100k_toolkit.classification.timm_trainer import (
            build_transforms,
            import_timm,
        )

        data: dict[str, Any] = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        self.model_name = data["model_name"]
        self.class_names = list(data["class_names"])
        model = import_timm().create_model(
            self.model_name, pretrained=False, num_classes=len(self.class_names)
        )
        model.load_state_dict(data["state_dict"])
        self._model = model.to(self._device).eval()
        _, self._transform = build_transforms(
            data["imgsz"], tuple(data["mean"]), tuple(data["std"])
        )

    def _load_ultralytics(self, checkpoint: str | Path) -> None:
        from ultralytics import YOLO

        self._yolo = YOLO(str(checkpoint))
        if self._yolo.task != "classify":
            raise ValueError(
                f"{checkpoint} is a '{self._yolo.task}' model, not a classifier"
            )
        names = self._yolo.names
        self.class_names = [names[i] for i in sorted(names)]
        yaml = getattr(self._yolo.model, "yaml", None)
        yaml_file = yaml.get("yaml_file", "") if isinstance(yaml, dict) else ""
        self.model_name = Path(yaml_file).stem or Path(checkpoint).parent.parent.name

    def predict(self, image: Image.Image) -> Prediction:
        """Classify one image (any mode; converted to RGB)."""
        image = image.convert("RGB")
        started = time.perf_counter()
        if self.backend == "timm":
            with torch.no_grad():
                batch = self._transform(image).unsqueeze(0).to(self._device)
                probabilities = self._model(batch).float().softmax(dim=1)[0].cpu()
        else:
            results: Any = self._yolo.predict(
                source=image, device=str(self._device), verbose=False
            )
            probs = results[0].probs
            if probs is None:
                raise RuntimeError("Ultralytics returned no class probabilities")
            probabilities = torch.as_tensor(probs.data).float().cpu()
        return Prediction(
            self.class_names,
            [float(p) for p in probabilities],
            time.perf_counter() - started,
        )


def check_task_classes(task: str, class_names: list[str]) -> None:
    """
    Raise if a checkpoint's classes do not belong to dataset ``bdd100k-<task>``.

    Catches pointing a weather checkpoint at ``period``. A checkpoint trained with
    ``--exclude-unknown`` (no ``unknown`` class) is fine.
    """
    from bdd100k_toolkit.classification import get_spec, list_datasets

    key = task if task.startswith("bdd100k-") else f"bdd100k-{task}"
    if key not in list_datasets():
        tasks = ", ".join(k.removeprefix("bdd100k-") for k in list_datasets())
        raise ValueError(f"Unknown task '{task}'. Choose from: {tasks}.")
    expected = set(get_spec(key).classes)
    unexpected = sorted(set(class_names) - expected)
    if unexpected:
        raise ValueError(
            f"This checkpoint predicts {class_names}, which does not match task "
            f"'{task}' (unexpected: {unexpected}). Is --task the right one?"
        )
