"""Tests for the classification dataset registry."""

from __future__ import annotations

import pytest

from bdd100k_toolkit.classification import get, get_spec, list_datasets


def test_list_datasets_includes_all_three() -> None:
    datasets = list_datasets()
    assert "bdd100k-period" in datasets
    assert "bdd100k-weather" in datasets
    assert "bdd100k-scenario" in datasets


def test_get_spec_returns_expected_class_counts() -> None:
    assert get_spec("bdd100k-period").num_classes == 3
    assert get_spec("bdd100k-weather").num_classes == 6
    assert get_spec("bdd100k-scenario").num_classes == 6


def test_get_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get("not-a-real-dataset")


def test_get_returns_adapter_instance() -> None:
    adapter = get("bdd100k-weather")
    assert hasattr(adapter, "prepare_classification")
    assert adapter.spec.key == "bdd100k-weather"
