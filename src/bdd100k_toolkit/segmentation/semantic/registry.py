"""Central registry mapping a semantic segmentation dataset key to its adapter."""

from __future__ import annotations

from typing import TypeVar

from bdd100k_toolkit.segmentation.semantic.base import (
    SemanticSegAdapter,
    SemanticSegSpec,
)

_REGISTRY: dict[str, type[SemanticSegAdapter]] = {}

AdapterT = TypeVar("AdapterT", bound=type[SemanticSegAdapter])


def register(adapter_cls: AdapterT) -> AdapterT:
    """Register a dataset adapter class under its ``spec.key`` (a class decorator)."""
    key = adapter_cls.spec.key
    if key in _REGISTRY:
        raise ValueError(f"Dataset '{key}' is already registered.")
    _REGISTRY[key] = adapter_cls
    return adapter_cls


def get(key: str) -> SemanticSegAdapter:
    """Instantiate the registered adapter for ``key``."""
    return _get_class(key)()


def get_spec(key: str) -> SemanticSegSpec:
    """Return the registered ``SemanticSegSpec`` for ``key``."""
    return _get_class(key).spec


def list_datasets() -> list[str]:
    """Return all registered dataset keys, sorted."""
    return sorted(_REGISTRY)


def _get_class(key: str) -> type[SemanticSegAdapter]:
    try:
        return _REGISTRY[key]
    except KeyError as err:
        supported = ", ".join(list_datasets())
        raise KeyError(f"Unknown dataset '{key}'. Supported: {supported}.") from err
