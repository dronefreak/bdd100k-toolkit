r"""
`bdd100k-evaluate`: evaluate a trained YOLO detector.

Computes standard YOLO object-detection metrics (mAP@0.5, mAP@0.5:0.95,
precision, recall, per-class breakdown) on a canonical Ultralytics data.yaml
split using the Ultralytics validation engine.

Usage:
  bdd100k-evaluate --checkpoint <run>/weights/best.pt \
      --dataset bdd100k-detection --dataset-yaml /path/to/yolo_dataset/data.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.table import Table

from bdd100k_toolkit.detection import get_spec, list_datasets
from bdd100k_toolkit.utils.console import RichConsoleManager


@dataclass(frozen=True)
class EvaluationOptions:
    """Store runtime options for YOLO detection evaluation."""

    checkpoint_path: str
    dataset_yaml: str | Path
    class_names: list[str]
    num_classes: int
    device: str
    output_dir: Path
    save_predictions: bool = False
    split: str = "test"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for detection evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a BDD100K-Toolkit YOLO detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint / .pt file"
    )
    parser.add_argument("--model", default="yolo11n", help="Model name")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list_datasets(),
        help="Registered dataset key (determines class names/count)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (defaults to the full class count for --dataset)",
    )
    parser.add_argument("--dataset-yaml", required=True, help="Ultralytics data.yaml")
    parser.add_argument(
        "--split", default="test", help="Dataset split to evaluate (default: test)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", default="eval_outputs", help="Output directory")
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save predictions JSON"
    )
    return parser.parse_args()


def evaluate_detection(options: EvaluationOptions) -> dict[str, Any]:
    """Evaluate a YOLO detection model using the Ultralytics val engine."""
    try:
        from ultralytics import YOLO as UltralyticsYOLO
    except ImportError as err:
        raise ImportError("pip install ultralytics>=8.0.0") from err

    console = RichConsoleManager.get_console()
    console.print(
        "\n[bold cyan]YOLO detection evaluation: Ultralytics val engine[/bold cyan]"
    )

    names = options.class_names[: min(options.num_classes, len(options.class_names))]

    model = UltralyticsYOLO(str(options.checkpoint_path))
    results = model.val(
        data=str(options.dataset_yaml),
        device=options.device,
        split=options.split,
        save_json=options.save_predictions,
        project=str(options.output_dir.resolve()),
        name="yolo_eval",
        exist_ok=True,
    )

    metrics: dict[str, Any] = {}
    if hasattr(results, "box"):
        metrics["mAP50"] = float(results.box.map50)
        metrics["mAP50_95"] = float(results.box.map)
        metrics["precision"] = float(results.box.mp)
        metrics["recall"] = float(results.box.mr)
        if (
            hasattr(results.box, "ap_class_index")
            and results.box.ap_class_index is not None
        ):
            metrics["per_class"] = {}
            for i, cls_idx in enumerate(results.box.ap_class_index):
                cls_name = (
                    names[cls_idx] if cls_idx < len(names) else f"class_{cls_idx}"
                )
                metrics["per_class"][cls_name] = {
                    "mAP50": float(results.box.ap50[i])
                    if i < len(results.box.ap50)
                    else 0.0,
                    "mAP50_95": float(results.box.ap[i])
                    if i < len(results.box.ap)
                    else 0.0,
                }

    return metrics


def print_metrics_table(model_name: str, metrics: dict[str, Any]) -> None:
    """Print a rich table of evaluation results."""
    console = RichConsoleManager.get_console()
    console.rule(f"[bold]Evaluation Results: {model_name}[/bold]")

    summary = Table(title="Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")

    def fmt(v: Any) -> str:
        if v is None:
            return "[dim]N/A[/dim]"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for key in ("mAP50", "mAP50_95", "precision", "recall"):
        if key in metrics:
            label = {"mAP50_95": "mAP@0.5:0.95", "mAP50": "mAP@0.5"}.get(
                key, key.title()
            )
            summary.add_row(label, fmt(metrics[key]))
    console.print(summary, style="bold green")

    per_class = metrics.get("per_class", {})
    if per_class:
        cls_table = Table(
            title="Per-Class Metrics", show_header=True, header_style="bold cyan"
        )
        cls_table.add_column("Class", style="white")
        cls_table.add_column("mAP@0.5", justify="right")
        cls_table.add_column("mAP@0.5:0.95", justify="right")
        for cls_name, cls_m in sorted(per_class.items()):
            cls_table.add_row(
                cls_name,
                f"{cls_m.get('mAP50', 0):.4f}",
                f"{cls_m.get('mAP50_95', 0):.4f}",
            )
        console.print(cls_table, style="bold green")


def finalize_metrics(
    model_name: str, metrics: dict[str, Any], output_dir: Path
) -> Path:
    """Print the results table, save metrics.json, and return its path."""
    console = RichConsoleManager.get_console()
    print_metrics_table(model_name, metrics)

    metrics_path = output_dir / "metrics.json"
    serializable: dict[str, Any] = {
        k: v for k, v in metrics.items() if k != "per_class"
    }
    if "per_class" in metrics:
        serializable["per_class"] = metrics["per_class"]
    metrics_path.write_text(json.dumps(serializable, indent=2))
    console.print(f"\n✓ Metrics saved to [bold]{metrics_path}[/bold]")
    return metrics_path


def main() -> None:
    """Run the detection evaluation CLI entrypoint."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = get_spec(args.dataset)
    num_classes = args.num_classes if args.num_classes is not None else spec.num_classes

    console = RichConsoleManager.get_console()
    console.print("\n[bold green]BDD100K-Toolkit Detection Evaluation[/bold green]")
    console.print(f"  Model: [bold]{args.model}[/bold]")
    console.print(f"  Dataset: {spec.display_name}")
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Device: {args.device}\n")

    metrics = evaluate_detection(
        EvaluationOptions(
            checkpoint_path=args.checkpoint,
            dataset_yaml=args.dataset_yaml,
            class_names=spec.classes,
            num_classes=num_classes,
            device=args.device,
            output_dir=output_dir,
            save_predictions=args.save_predictions,
            split=args.split,
        )
    )
    finalize_metrics(args.model, metrics, output_dir)


if __name__ == "__main__":
    main()
