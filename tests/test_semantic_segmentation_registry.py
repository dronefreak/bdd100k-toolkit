"""Tests for the semantic segmentation dataset registry."""

from __future__ import annotations

import pytest

from bdd100k_toolkit.segmentation.semantic import get, get_spec, list_datasets


def test_bdd100k_semantic_seg_is_registered() -> None:
    assert "bdd100k-semantic-seg" in list_datasets()


def test_get_spec_returns_19_classes() -> None:
    spec = get_spec("bdd100k-semantic-seg")
    assert spec.num_classes == 19
    assert spec.ignore_index == 255


def test_get_returns_adapter_instance() -> None:
    adapter = get("bdd100k-semantic-seg")
    assert adapter.spec.key == "bdd100k-semantic-seg"


def test_get_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get("not-a-real-dataset")
