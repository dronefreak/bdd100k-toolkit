r"""
`bdd100k-evaluate`: evaluate a trained classifier.

Evaluates a checkpoint on the canonical ``test`` split.

Usage:
  bdd100k-evaluate --dataset bdd100k-weather \
      --checkpoint <run>/weights/best.pt --data-dir /path/to/canonical_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bdd100k_toolkit.classification import list_datasets
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
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto (default)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write metrics.json (default: alongside checkpoint)",
    )
    return parser.parse_args()


def main() -> None:
    """Run classification evaluation and print/save top1/top5 accuracy."""
    from ultralytics import YOLO

    args = parse_args()
    console = RichConsoleManager.get_console()
    console.print(
        "\n[bold green]BDD100K-Toolkit Classification Evaluation[/bold green]"
    )
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Data dir: {args.data_dir}\n")

    model = YOLO(args.checkpoint)
    metrics = model.val(
        data=str(Path(args.data_dir).resolve()),
        split=args.split,
        device=resolve_device(args.device),
    )

    results = {
        "top1_accuracy": float(metrics.top1),
        "top5_accuracy": float(metrics.top5),
    }
    console.print(f"  Top-1 accuracy: {results['top1_accuracy']:.4f}")
    console.print(f"  Top-5 accuracy: {results['top5_accuracy']:.4f}")

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
