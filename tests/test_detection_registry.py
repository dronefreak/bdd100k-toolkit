"""Tests for the detection dataset registry."""

from __future__ import annotations

import pytest

from bdd100k_toolkit.detection import get, get_spec, list_datasets


def test_list_datasets_includes_bdd100k_detection() -> None:
    assert "bdd100k-detection" in list_datasets()


def test_get_spec_returns_expected_class_count() -> None:
    assert get_spec("bdd100k-detection").num_classes == 10


def test_get_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get("not-a-real-dataset")


def test_get_returns_adapter_instance() -> None:
    adapter = get("bdd100k-detection")
    assert hasattr(adapter, "prepare_coco")
    assert adapter.spec.key == "bdd100k-detection"
