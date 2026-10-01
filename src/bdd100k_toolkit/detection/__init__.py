"""Detection task package: adapters, registry, and Ultralytics trainer."""

from bdd100k_toolkit.detection import datasets  # noqa: F401
from bdd100k_toolkit.detection.registry import get, get_spec, list_datasets

__all__ = ["get", "get_spec", "list_datasets"]
