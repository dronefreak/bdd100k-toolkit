"""
Mean Intersection-over-Union (mIoU) computation, dependency-free.

Standard semantic segmentation metric; implemented directly with ``numpy``
rather than pulling in a metrics library, since the confusion-matrix-based
computation is a handful of lines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

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


def category_confusion(
    confusion: np.ndarray,
    classes: Sequence[str],
    categories: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """
    Sum a class confusion matrix into a category one (rows true, columns predicted).

    Confusing two classes of the same category counts as correct, as in the
    Cityscapes category-level IoU.
    """
    index = {name: i for i, name in enumerate(classes)}
    groups = np.zeros((len(classes), len(categories)), dtype=np.int64)
    for column, members in enumerate(categories.values()):
        for member in members:
            groups[index[member], column] = 1
    return groups.T @ confusion.astype(np.int64) @ groups


def segmentation_report(
    confusion: np.ndarray,
    classes: Sequence[str],
    categories: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """
    Summarise a dataset-wide ``confusion[true, predicted]`` matrix.

    ``miou`` follows the official BDD100K evaluator: the mean over the classes
    that occur in the ground truth, with IoU 0 for a class nothing matched.
    ``fiou`` weights each IoU by the class's share of ground-truth pixels and
    ``pacc`` is pixel accuracy. Precision and recall are per class
    (``diag / predicted`` and ``diag / ground truth``). The matrix must not
    contain ignored pixels (see :func:`confusion_matrix`).
    """
    confusion = confusion.astype(np.int64)
    diagonal = np.diag(confusion).astype(np.float64)
    truth, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    total = max(int(confusion.sum()), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.nan_to_num(diagonal / (truth + predicted - diagonal))
        precision = np.nan_to_num(diagonal / predicted)
        recall = np.nan_to_num(diagonal / truth)
    present = truth > 0
    report: dict[str, Any] = {
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "fiou": float((iou * truth / total).sum()),
        "pacc": float(diagonal.sum() / total),
        "per_class": {
            name: {
                "iou": float(iou[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "pixels": int(truth[i]),
            }
            for i, name in enumerate(classes)
        },
    }
    if categories:
        category_iou = segmentation_report(
            category_confusion(confusion, classes, categories), list(categories)
        )
        report["category_miou"] = category_iou["miou"]
        report["category_iou"] = {
            name: values["iou"] for name, values in category_iou["per_class"].items()
        }
    return report
