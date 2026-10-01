"""
Base types for the classification dataset registry.

An adapter's only job is to convert a raw dataset download into the
canonical layout: ``output_dir/{train,valid,test}/<class_name>/*.jpg``: a
plain ``torchvision.datasets.ImageFolder``-compatible directory tree.
Deliberately chosen because Ultralytics' classification trainer
(``yolo11n-cls`` et al.) already reads this exact layout with zero extra
glue, the same way DetectionBench's canonical COCO layout lets one converter
drive both Ultralytics and RF-DETR without either framework leaking into the
adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from bdd100k_toolkit.utils.io import link_image

__all__ = [
    "ClassificationAdapter",
    "ClassificationSpec",
    "link_image",
]


@dataclass(frozen=True)
class ClassificationSpec:
    """Metadata describing a supported classification dataset."""

    key: str
    display_name: str
    classes: list[str]  # canonical class folder names, incl. ``unknown``
    description: str | None = None
    homepage: str | None = None
    citation: str | None = None
    license: str | None = None

    @property
    def num_classes(self) -> int:
        """Number of classes in this dataset."""
        return len(self.classes)


class ClassificationAdapter(ABC):
    """Converts a raw dataset download into the canonical ImageFolder layout."""

    spec: ClassVar[ClassificationSpec]

    @abstractmethod
    def prepare_classification(
        self,
        raw_dir: Path,
        output_dir: Path,
        *,
        include_unknown: bool = True,
        labels_dir: Path | None = None,
    ) -> None:
        """
        Convert ``raw_dir`` (a raw dataset download) into canonical splits.

        Args:
            raw_dir: Download root (official images + labels, or Kaggle folders).
            output_dir: Canonical output root.
            include_unknown: Keep the ``unknown`` class (official ``undefined``).
            labels_dir: Label JSON folder when it is not ``raw_dir/labels``.

        """
