r"""
`bdd100k-evaluate`: evaluate a trained classifier.

Evaluates a checkpoint on a canonical split (``test`` by default) and reports
top-1 / top-5 accuracy plus per-class precision, recall and F1, macro averages
and the confusion matrix. The BDD100K attribute tasks are heavily imbalanced,
so the macro metrics are the ones to compare models with.

Usage:
  bdd100k-evaluate --dataset bdd100k-weather \
      --checkpoint <run>/weights/best.pt --data-dir /path/to/canonical_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.table import Table

from bdd100k_toolkit.classification import list_datasets
from bdd100k_toolkit.utils.cls_metrics import classification_metrics
from bdd100k_toolkit.utils.console import RichConsoleManager
from bdd100k_toolkit.utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a BDD100K-Toolkit classifier."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pt")
    parser.add_argument(
        "--dataset",
        choices=list_datasets(),
        default=None,
        help="Registered dataset key (optional; used by bdd100k-evaluate to "
        "pick the task)",
    )
    parser.add_argument(
        "--data-dir", required=True, help="Canonical classification root (has test/)"
    )
    parser.add_argument(
        "--split", default="test", help="Split to evaluate (default: test)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ultralytics", "timm"],
        default="auto",
        help="Checkpoint type (auto detects timm checkpoints; default: auto)",
    )
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto (default)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write metrics.json (default: alongside checkpoint)",
    )
    return parser.parse_args()


def build_report(
    top1: float,
    top5: float,
    ultralytics_confusion: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Combine Ultralytics' accuracy numbers with our per-class metrics.

    Ultralytics' classification confusion matrix is ``[predicted, true]``; ours
    is ``[true, predicted]``, so it is transposed here.
    """
    return report_from_confusion(
        np.asarray(ultralytics_confusion).T, top1, top5, class_names
    )


def report_from_confusion(
    confusion: np.ndarray, top1: float, top5: float, class_names: list[str]
) -> dict[str, Any]:
    """Build the metrics report from a ``confusion[true, predicted]`` matrix."""
    report = classification_metrics(confusion, class_names)
    report["top1_accuracy"] = top1
    report["top5_accuracy"] = top5
    report["class_names"] = class_names
    report["confusion_matrix"] = confusion.astype(int).tolist()
    return report


def print_report(report: dict[str, Any]) -> None:
    """Print summary and per-class tables."""
    console = RichConsoleManager.get_console()
    console.print(
        f"  Top-1 accuracy:    {report['top1_accuracy']:.4f}\n"
        f"  Balanced accuracy: {report['balanced_accuracy']:.4f}\n"
        f"  Macro precision:   {report['macro_precision']:.4f}\n"
        f"  Macro recall:      {report['macro_recall']:.4f}\n"
        f"  Macro F1:          {report['macro_f1']:.4f}"
    )
    table = Table(title="Per-class", header_style="bold cyan")
    table.add_column("Class")
    for column in ("Precision", "Recall", "F1", "Support"):
        table.add_column(column, justify="right")
    for name, values in report["per_class"].items():
        table.add_row(
            name,
            f"{values['precision']:.3f}",
            f"{values['recall']:.3f}",
            f"{values['f1']:.3f}",
            str(values["support"]),
        )
    console.print(table)


def _evaluate_ultralytics(args: argparse.Namespace, device: str) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(args.checkpoint)
    metrics = model.val(
        data=str(Path(args.data_dir).resolve()),
        split=args.split,
        device=device,
        plots=False,
    )
    class_names = [model.names[i] for i in sorted(model.names)]
    return build_report(
        float(metrics.top1),
        float(metrics.top5),
        np.asarray(metrics.confusion_matrix.matrix),
        class_names,
    )


def _evaluate_timm(args: argparse.Namespace, device: str) -> dict[str, Any]:
    from bdd100k_toolkit.classification.timm_trainer import evaluate_timm_checkpoint

    confusion, top5, class_names = evaluate_timm_checkpoint(
        args.checkpoint, args.data_dir, split=args.split, device=device
    )
    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    return report_from_confusion(confusion, accuracy, top5, class_names)


def main() -> None:
    """Run classification evaluation and print/save the metrics."""
    args = parse_args()
    console = RichConsoleManager.get_console()
    console.print(
        "\n[bold green]BDD100K-Toolkit Classification Evaluation[/bold green]"
    )
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Data dir: {args.data_dir} (split: {args.split})\n")

    device = resolve_device(args.device)
    if args.backend == "auto":
        from bdd100k_toolkit.classification.timm_trainer import is_timm_checkpoint

        backend = "timm" if is_timm_checkpoint(args.checkpoint) else "ultralytics"
    else:
        backend = args.backend
    console.print(f"  Backend: {backend}\n")

    if backend == "timm":
        report = _evaluate_timm(args, device)
    else:
        report = _evaluate_ultralytics(args, device)
    print_report(report)

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    console.print(f"\nMetrics saved to [bold]{output_dir / 'metrics.json'}[/bold]")


if __name__ == "__main__":
    main()
