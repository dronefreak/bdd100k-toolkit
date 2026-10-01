"""
Base types for the detection dataset registry.

An adapter's only job is to convert a raw dataset download into the
canonical COCO split layout:
``output_dir/{train,valid,test}/_annotations.coco.json`` plus the
corresponding images. Everything downstream (the COCO<->YOLO bridge
converter, training, evaluation) operates on that canonical layout, so an
adapter never needs to know anything about YOLO or Ultralytics, only its
own dataset's raw format. This mirrors DetectionBench's
``datasets/base.py``/``registry.py`` split exactly (this project only ever
registers one detection dataset (BDD100K itself) but keeps the same
adapter/registry shape so a second driving dataset, or a future
DetectionBench parity check, is a drop-in).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from bdd100k_toolkit.utils.io import link_image

__all__ = [
    "COCO_ANNOTATION_FILENAME",
    "DatasetAdapter",
    "DatasetSpec",
    "link_image",
]

COCO_ANNOTATION_FILENAME = "_annotations.coco.json"


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata describing a supported detection dataset."""

    key: str
    display_name: str
    classes: list[str]
    description: str | None = None
    homepage: str | None = None
    citation: str | None = None
    license: str | None = None

    @property
    def num_classes(self) -> int:
        """Number of detection classes in this dataset."""
        return len(self.classes)


class DatasetAdapter(ABC):
    """Converts a raw dataset download into the canonical COCO layout."""

    spec: ClassVar[DatasetSpec]

    @abstractmethod
    def prepare_coco(
        self, raw_dir: Path, output_dir: Path, *, labels_dir: Path | None = None
    ) -> None:
        """
        Convert ``raw_dir`` (a raw dataset download) into canonical COCO splits.

        ``labels_dir`` overrides ``raw_dir/labels`` when the labels were
        extracted separately from the images.
        """
