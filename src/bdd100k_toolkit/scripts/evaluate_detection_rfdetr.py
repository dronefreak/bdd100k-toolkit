r"""
`bdd100k-evaluate --model rfdetr-*`: evaluate an RF-DETR detector.

Runs the checkpoint over a canonical COCO split and scores it with
Supervision's mAP. Precision and recall need a real confidence cutoff (RF-DETR
emits a fixed number of low-confidence queries per image, which swamps
precision at threshold 0), so they are reported at the cutoff that maximises
F1 at IoU 0.5, like the Ultralytics validator does. The output keys match
``evaluate_detection`` so both share the table and ``metrics.json`` writer.

Usage:
  bdd100k-evaluate --dataset bdd100k-detection --model rfdetr-nano \
      --checkpoint <run>/checkpoint_best_total.pth --dataset-dir /path/to/coco_dataset
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from bdd100k_toolkit.detection import get_spec, list_datasets
from bdd100k_toolkit.detection.base import COCO_ANNOTATION_FILENAME
from bdd100k_toolkit.detection.rfdetr import (
    load_model_class,
    normalize_model_name,
    read_training_resolution,
)
from bdd100k_toolkit.scripts.evaluate_detection import finalize_metrics
from bdd100k_toolkit.utils.console import RichConsoleManager
from bdd100k_toolkit.utils.device import resolve_device

PR_THRESHOLDS = tuple(round(float(t), 2) for t in np.arange(0.05, 1.0, 0.05))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for RF-DETR evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a BDD100K-Toolkit RF-DETR detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = parser.add_argument
    add("--checkpoint", required=True, help="checkpoint_best_total.pth")
    add("--model", default="rfdetr-nano", help="rfdetr-nano | small | medium | large")
    add("--dataset", required=True, choices=list_datasets(), help="Dataset key")
    add("--dataset-dir", required=True, help="Canonical COCO root (train/valid/test)")
    add("--split", default="test", help="Split to evaluate")
    add("--device", default="auto", help="cuda / cpu / auto")
    add("--resolution", type=int, help="Default: read from training_config.json")
    add("--limit", type=int, help="Only evaluate the first N images (smoke test)")
    add("--output-dir", default="eval_outputs", help="Output directory")
    return parser.parse_args(argv)


def load_split(dataset_dir: Path, split: str) -> Any:
    """Load a COCO split as a Supervision detection dataset."""
    import supervision as sv

    split_dir = dataset_dir / split
    annotations = split_dir / COCO_ANNOTATION_FILENAME
    if not annotations.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotations}")
    return sv.DetectionDataset.from_coco(
        images_directory_path=str(split_dir), annotations_path=str(annotations)
    )


def best_f1_operating_point(
    predictions: list[Any], targets: list[Any]
) -> tuple[float, float, float]:
    """Return ``(threshold, precision, recall)`` maximising F1 at IoU 0.5."""
    from supervision.metrics import Precision, Recall

    best = (PR_THRESHOLDS[0], 0.0, 0.0)
    best_f1 = -1.0
    for threshold in PR_THRESHOLDS:
        precision_metric, recall_metric = Precision(), Recall()
        for pred, target in zip(predictions, targets, strict=True):
            kept = pred[pred.confidence >= threshold]
            precision_metric.update(kept, target)
            recall_metric.update(kept, target)
        p = float(precision_metric.compute().precision_at_50)
        r = float(recall_metric.compute().recall_at_50)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if f1 > best_f1:
            best_f1, best = f1, (threshold, p, r)
    return best


def evaluate_rfdetr(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate an RF-DETR checkpoint and return metrics in the YOLO key layout."""
    import cv2
    from supervision.metrics import MeanAveragePrecision

    console = RichConsoleManager.get_console()
    kwargs: dict[str, Any] = {
        "device": resolve_device(args.device),
        "pretrain_weights": str(Path(args.checkpoint).resolve()),
        "num_classes": get_spec(args.dataset).num_classes,
    }
    resolution = args.resolution or read_training_resolution(args.checkpoint)
    if resolution:
        kwargs["resolution"] = resolution
        console.print(f"  Resolution: {resolution}")
    else:
        console.print(
            "[yellow]No training_config.json next to the checkpoint: using the model "
            "family default resolution (pass --resolution if it differs).[/yellow]"
        )
    model = load_model_class(args.model)(**kwargs)

    dataset = load_split(Path(args.dataset_dir), args.split)
    total = min(len(dataset), args.limit) if args.limit else len(dataset)
    console.print(
        f"[bold cyan]Evaluating {normalize_model_name(args.model)} on "
        f"{args.split} ({total} images)[/bold cyan]"
    )

    map_metric = MeanAveragePrecision()
    predictions: list[Any] = []
    targets: list[Any] = []
    started = time.perf_counter()
    for index, (_, image, truth) in enumerate(dataset):
        if index >= total:
            break
        # Supervision loads BGR via cv2; RF-DETR only converts PIL inputs to RGB.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = model.predict(rgb, threshold=0.0, include_source_image=False)
        map_metric.update(detections, truth)
        predictions.append(detections)
        targets.append(truth)
    console.print(f"  inference took {time.perf_counter() - started:.1f}s")

    result = map_metric.compute()
    threshold, precision, recall = best_f1_operating_point(predictions, targets)

    categories = json.loads(
        (Path(args.dataset_dir) / args.split / COCO_ANNOTATION_FILENAME).read_text()
    )["categories"]
    names = {int(c["id"]): str(c["name"]) for c in categories}
    per_class = {
        names.get(int(cls), str(cls)): {
            "mAP50": float(ap[0]),
            "mAP50_95": float(np.mean(ap)),
        }
        for cls, ap in zip(result.matched_classes, result.ap_per_class, strict=True)
    }
    metrics: dict[str, Any] = {
        "mAP50": float(result.map50),
        "mAP50_95": float(result.map50_95),
        "mAP75": float(result.map75),
        "precision": precision,
        "recall": recall,
        "pr_confidence_threshold": threshold,
        "small_objects_mAP50_95": _optional(result.small_objects),
        "medium_objects_mAP50_95": _optional(result.medium_objects),
        "large_objects_mAP50_95": _optional(result.large_objects),
        "split": args.split,
        "num_images": total,
        "per_class": per_class,
    }
    return metrics


def _optional(scale: Any) -> float | None:
    return float(scale.map50_95) if scale is not None else None


def main(argv: list[str] | None = None) -> None:
    """Run the RF-DETR evaluation CLI entrypoint."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_rfdetr(args)
    finalize_metrics(args.model, metrics, output_dir)


if __name__ == "__main__":
    main()
