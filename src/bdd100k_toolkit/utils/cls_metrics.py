"""
Classification metrics from a confusion matrix, dependency-free (numpy only).

The BDD100K attribute tasks are heavily imbalanced (e.g. 7 ``gas stations``
and 13 ``foggy`` images in the test split against thousands of ``clear`` /
``city street`` ones), so top-1 accuracy alone says little. These metrics
report per-class precision / recall / F1 and the macro averages, which treat
every class equally.

Convention: ``confusion[true, predicted]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division returning 0 where the denominator is 0."""
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def classification_metrics(
    confusion: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """
    Compute accuracy and per-class / macro precision, recall and F1.

    Args:
        confusion: ``(n, n)`` integer matrix, ``confusion[true, predicted]``.
        class_names: The ``n`` class names, in matrix order.

    Returns:
        dict with ``accuracy``, ``balanced_accuracy`` (macro recall),
        ``macro_precision``, ``macro_recall``, ``macro_f1`` and ``per_class``
        (``{name: {precision, recall, f1, support}}``). Macro averages are taken
        over classes that have at least one true sample, so a class absent from
        the evaluated split does not drag the average to zero.

    """
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError(
            f"confusion matrix is {matrix.shape} but there are "
            f"{len(class_names)} classes"
        )
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    present = support > 0
    total = matrix.sum()
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }


# Metrics a trainer may monitor to pick the best checkpoint and to early-stop.
# All are "higher is better".
MONITORS = ("macro_f1", "balanced_accuracy", "accuracy")
DEFAULT_MONITOR = "macro_f1"


def check_monitor(monitor: str) -> str:
    """Return ``monitor`` if it is a supported metric name, else raise."""
    if monitor not in MONITORS:
        raise ValueError(f"monitor must be one of {MONITORS}, got {monitor!r}")
    return monitor


def monitor_value(confusion: np.ndarray, monitor: str) -> float:
    """Return the ``monitor`` metric from a ``confusion[true, predicted]`` matrix."""
    check_monitor(monitor)
    names = [str(i) for i in range(len(confusion))]
    return float(classification_metrics(confusion, names)[monitor])
