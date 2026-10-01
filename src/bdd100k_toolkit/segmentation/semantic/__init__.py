"""Semantic segmentation task package: adapters, registry, and trainer."""

from bdd100k_toolkit.segmentation.semantic import datasets  # noqa: F401
from bdd100k_toolkit.segmentation.semantic.registry import get, get_spec, list_datasets

__all__ = ["get", "get_spec", "list_datasets"]
