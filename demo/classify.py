#!/usr/bin/env python
r"""
Classify one image with trained BDD100K attribute classifiers and draw the result.

One ``--model TASK=CHECKPOINT`` shows that task; several show them all. Works with
checkpoints from either training backend (detected from the file).

By default the predictions are drawn as an overlay on the image. With
``--generate-dashboard`` you get a 1920x1080 dashboard instead: the frame, one card
per task with all class probabilities, and a strip with models and latency.

  python demo/classify.py --image frame.jpg \
      --model weather=experiments/weather/yolo11n-cls/weights/best.pt \
      --model period=experiments/period/resnet18/weights/best.pt \
      --model scenario=experiments/scenario/resnet18/weights/best.pt

The picture is saved to demo/outputs/ (or --output) and a table is printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps
from rich.console import Console
from rich.table import Table

from bdd100k_toolkit.classification import list_datasets
from bdd100k_toolkit.classification.predict import (
    BACKENDS,
    Predictor,
    check_task_classes,
)
from bdd100k_toolkit.utils.dashboard import render_dashboard
from bdd100k_toolkit.utils.draw import TaskResult
from bdd100k_toolkit.utils.overlay import render_overlay
from bdd100k_toolkit.utils.timing import median_prediction

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TASKS = [key.removeprefix("bdd100k-") for key in list_datasets()]


def parse_model(value: str) -> tuple[str, str]:
    """Parse ``TASK=CHECKPOINT`` (the ``--model`` argument type)."""
    task, _, checkpoint = value.partition("=")
    if task not in TASKS or not checkpoint:
        raise argparse.ArgumentTypeError(f"expected TASK=CHECKPOINT, TASK in {TASKS}")
    return task, checkpoint


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add = parser.add_argument
    add("--image", required=True, help="Image to classify")
    add(
        "--model",
        action="append",
        required=True,
        type=parse_model,
        metavar="TASK=CHECKPOINT",
        help=f"Checkpoint for a task ({', '.join(TASKS)})",
    )
    add("--backend", choices=BACKENDS, default="auto", help="Checkpoint type")
    add("--device", default="auto", help="cuda / cpu / auto")
    add("--runs", type=int, default=5, help="Timed runs; the median is shown")
    add(
        "--generate-dashboard",
        action="store_true",
        help="Draw a 1920x1080 dashboard instead of the overlay",
    )
    add("--output", help=f"Output picture (default {OUTPUT_DIR}/<image>.jpg)")
    args = parser.parse_args(argv)
    tasks = [task for task, _ in args.model]
    if len(set(tasks)) != len(tasks):
        parser.error("give each task at most once")
    return args


def run_models(args: argparse.Namespace, image: Image.Image) -> list[TaskResult]:
    """Load each checkpoint, check it matches its task, and predict on ``image``."""
    results = []
    for task, checkpoint in args.model:
        predictor = Predictor(checkpoint, args.backend, args.device)
        check_task_classes(task, predictor.class_names)
        prediction = median_prediction(predictor, image, args.runs)
        results.append(
            TaskResult(task, predictor.model_name, predictor.backend, prediction)
        )
    return results


def print_results(console: Console, name: str, results: list[TaskResult]) -> None:
    """Print one line per task."""
    table = Table(title=name, header_style="bold cyan")
    for column in ("task", "prediction", "confidence", "model", "latency"):
        table.add_column(column)
    for task, model, backend, prediction in results:
        label, top = prediction.top1
        ms = f"{prediction.seconds * 1000:.1f} ms"
        table.add_row(task, label, f"{top:.1%}", f"{model} ({backend})", ms)
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    """Run the demo; return a process exit code."""
    args = parse_args(argv)
    console, path = Console(), Path(args.image)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        with console.status("running models"):
            results = run_models(args, image)
    except (FileNotFoundError, ValueError) as err:
        Console(stderr=True).print(f"[red]error:[/red] {err}")
        return 2

    suffix = "_dashboard" if args.generate_dashboard else ""
    output = Path(args.output or OUTPUT_DIR / f"{path.stem}{suffix}.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.generate_dashboard:
        render_dashboard(image, path.name, results).save(output, quality=92)
    else:
        render_overlay(image, results).save(output, quality=92)
    print_results(console, path.name, results)
    console.print(f"saved [bold]{output}[/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
