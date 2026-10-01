"""
Mean Intersection-over-Union (mIoU) computation, dependency-free.

Standard semantic segmentation metric; implemented directly with ``numpy``
rather than pulling in a metrics library, since the confusion-matrix-based
computation is a handful of lines.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class MeanIoU(TypedDict):
    """Result of :func:`mean_iou`."""

    miou: float
    per_class_iou: list[float]


def confusion_matrix(
    pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int
) -> np.ndarray:
    """Accumulate a ``num_classes x num_classes`` confusion matrix for one pair."""
    valid = target != ignore_index
    pred, target = pred[valid], target[valid]
    indices = num_classes * target.astype(np.int64) + pred.astype(np.int64)
    counts = np.bincount(indices, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)


def mean_iou(confusion: np.ndarray) -> MeanIoU:
    """
    Compute per-class IoU and mIoU from an accumulated confusion matrix.

    Returns:
        dict with 'miou' (float, averaged over classes with any ground-truth
        or predicted pixels) and 'per_class_iou' (list, one entry per class;
        ``nan`` for classes absent from both prediction and ground truth).

    """
    intersection = np.diag(confusion).astype(np.float64)
    union = (
        confusion.sum(axis=0).astype(np.float64)
        + confusion.sum(axis=1).astype(np.float64)
        - intersection
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_iou = np.where(union > 0, intersection / union, np.nan)
    valid_classes = ~np.isnan(per_class_iou)
    miou = float(np.mean(per_class_iou[valid_classes])) if valid_classes.any() else 0.0
    return {"miou": miou, "per_class_iou": per_class_iou.tolist()}
