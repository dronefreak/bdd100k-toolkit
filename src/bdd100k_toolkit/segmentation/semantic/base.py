"""
Base types for the semantic segmentation dataset registry.

An adapter's only job is to convert a raw dataset download into the
canonical layout: ``output_dir/{train,valid,test}/images/*.jpg`` +
``output_dir/{train,valid,test}/masks/*.png``: one 1-channel PNG mask per
image, pixel value = class id (``255`` = ignore). This is the same plain
mask-pair layout most segmentation libraries (e.g.
``segmentation-models-pytorch``, ``mmsegmentation``) read directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from bdd100k_toolkit.utils.io import link_image

__all__ = [
    "IGNORE_INDEX",
    "SemanticSegAdapter",
    "SemanticSegSpec",
    "link_image",
]

# BDD100K's official "unknown"/ignore pixel value for semantic segmentation.
IGNORE_INDEX = 255


@dataclass(frozen=True)
class SemanticSegSpec:
    """Metadata describing a supported semantic segmentation dataset."""

    key: str
    display_name: str
    classes: list[str]
    ignore_index: int = IGNORE_INDEX
    description: str | None = None
    homepage: str | None = None
    citation: str | None = None
    license: str | None = None

    @property
    def num_classes(self) -> int:
        """Number of (non-ignored) semantic classes in this dataset."""
        return len(self.classes)


class SemanticSegAdapter(ABC):
    """Converts a raw dataset download into the canonical image/mask layout."""

    spec: ClassVar[SemanticSegSpec]

    @abstractmethod
    def prepare_segmentation(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert ``raw_dir`` (a raw dataset download) into canonical splits."""
