"""Tests for ``bdd100k_toolkit.utils.miou``."""

from __future__ import annotations

import numpy as np

from bdd100k_toolkit.utils.miou import confusion_matrix, mean_iou


def test_confusion_matrix_perfect_prediction() -> None:
    target = np.array([0, 0, 1, 1, 2, 2])
    pred = target.copy()
    cm = confusion_matrix(pred, target, num_classes=3, ignore_index=255)
    assert np.array_equal(np.diag(cm), [2, 2, 2])
    assert cm.sum() == 6


def test_confusion_matrix_excludes_ignore_index() -> None:
    target = np.array([0, 255, 1])
    pred = np.array([0, 1, 1])
    cm = confusion_matrix(pred, target, num_classes=2, ignore_index=255)
    assert cm.sum() == 2  # the ignore_index=255 pixel is dropped


def test_mean_iou_perfect_prediction_is_one() -> None:
    target = np.array([0, 0, 1, 1])
    pred = target.copy()
    cm = confusion_matrix(pred, target, num_classes=2, ignore_index=255)
    metrics = mean_iou(cm)
    assert metrics["miou"] == 1.0
    assert metrics["per_class_iou"] == [1.0, 1.0]


def test_mean_iou_absent_class_is_nan() -> None:
    target = np.array([0, 0, 0])
    pred = np.array([0, 0, 0])
    cm = confusion_matrix(pred, target, num_classes=3, ignore_index=255)
    metrics = mean_iou(cm)
    # class 1 and 2 never appear in prediction or ground truth -> nan, excluded.
    assert metrics["miou"] == 1.0
    assert metrics["per_class_iou"][0] == 1.0
    assert all(np.isnan(v) for v in metrics["per_class_iou"][1:])


def test_mean_iou_partial_overlap() -> None:
    target = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    cm = confusion_matrix(pred, target, num_classes=2, ignore_index=255)
    metrics = mean_iou(cm)
    # class 0: intersection=1, union=2 -> 0.5; class 1: intersection=2, union=3 -> 0.667
    assert metrics["per_class_iou"][0] == 0.5
    assert round(metrics["per_class_iou"][1], 3) == 0.667
