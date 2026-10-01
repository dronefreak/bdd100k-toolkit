"""Tests for ``bdd100k_toolkit.utils.cls_metrics``."""

from __future__ import annotations

import numpy as np
import pytest

from bdd100k_toolkit.utils.cls_metrics import classification_metrics


def test_perfect_prediction() -> None:
    metrics = classification_metrics(np.diag([5, 3]), ["a", "b"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["a"]["support"] == 5


def test_known_values_rows_are_true_classes() -> None:
    # true a: 8 right, 2 predicted b; true b: 1 predicted a, 9 right
    confusion = np.array([[8, 2], [1, 9]])
    metrics = classification_metrics(confusion, ["a", "b"])
    assert metrics["accuracy"] == pytest.approx(17 / 20)
    a = metrics["per_class"]["a"]
    assert a["recall"] == pytest.approx(8 / 10)
    assert a["precision"] == pytest.approx(8 / 9)
    assert a["f1"] == pytest.approx(2 * (8 / 9) * (8 / 10) / (8 / 9 + 8 / 10))
    assert metrics["balanced_accuracy"] == pytest.approx((0.8 + 0.9) / 2)


def test_majority_class_predictor_has_high_accuracy_but_low_macro_f1() -> None:
    confusion = np.array([[100, 0], [10, 0]])  # always predicts class a
    metrics = classification_metrics(confusion, ["common", "rare"])
    assert metrics["accuracy"] > 0.9
    assert metrics["per_class"]["rare"]["f1"] == 0.0
    assert metrics["macro_f1"] < 0.5


def test_class_absent_from_split_is_excluded_from_macro_average() -> None:
    confusion = np.array([[4, 0, 0], [0, 6, 0], [0, 0, 0]])
    metrics = classification_metrics(confusion, ["a", "b", "never"])
    assert metrics["macro_recall"] == 1.0
    assert metrics["per_class"]["never"]["support"] == 0


def test_predicted_but_never_true_class_gets_zero_precision_not_nan() -> None:
    confusion = np.array([[3, 1], [0, 0]])
    metrics = classification_metrics(confusion, ["a", "b"])
    assert metrics["per_class"]["b"]["precision"] == 0.0
    assert not np.isnan(metrics["macro_f1"])


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="classes"):
        classification_metrics(np.zeros((2, 2)), ["a", "b", "c"])


def test_empty_matrix_is_zero() -> None:
    assert classification_metrics(np.zeros((2, 2)), ["a", "b"])["accuracy"] == 0.0


def test_build_report_transposes_ultralytics_confusion() -> None:
    from bdd100k_toolkit.scripts.evaluate_classification import build_report

    # Ultralytics layout [predicted, true]: 2 true "a" predicted as "b".
    ultralytics = np.array([[8, 1], [2, 9]])
    report = build_report(0.85, 1.0, ultralytics, ["a", "b"])
    assert report["confusion_matrix"] == [[8, 2], [1, 9]]  # [true, predicted]
    assert report["per_class"]["a"]["recall"] == pytest.approx(0.8)
    assert report["top1_accuracy"] == 0.85
    assert report["class_names"] == ["a", "b"]
