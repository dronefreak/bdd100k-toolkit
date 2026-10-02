#!/usr/bin/env python
r"""
Classify images or videos with trained BDD100K attribute classifiers.

One ``--model TASK=CHECKPOINT`` shows that task; several show them all. Works with
checkpoints from either training backend (detected from the file).

By default the predictions are drawn as an overlay on the image. With
``--generate-dashboard`` you get a 1920x1080 dashboard instead: the frame, one card
per task with all class probabilities, and a strip with models and latency.

  python demo/classify.py --image frame.jpg \
      --model weather=experiments/weather/yolo11n-cls/weights/best.pt \
      --model period=experiments/period/resnet18/weights/best.pt \
      --model scenario=experiments/scenario/resnet18/weights/best.pt

--image can also be a folder (jpg, png, bmp, gif, tiff, webp, ... directly inside it).
--video takes a video file or a folder of videos (mp4, mov, avi, mkv, ...); every
--stride-th frame is classified (the others reuse the last result) and an annotated
mp4 is written. Outputs go to demo/outputs/ (or --output: a file for a single input,
else a folder).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.progress import track
from rich.table import Table

from bdd100k_toolkit.classification import list_datasets
from bdd100k_toolkit.classification.predict import (
    BACKENDS,
    Predictor,
    check_task_classes,
)
from bdd100k_toolkit.utils.dashboard import render_dashboard
from bdd100k_toolkit.utils.draw import TaskResult
from bdd100k_toolkit.utils.images import list_media, load_image
from bdd100k_toolkit.utils.overlay import render_overlay
from bdd100k_toolkit.utils.timing import median_prediction
from bdd100k_toolkit.utils.video import VIDEO_EXTENSIONS, VideoWriter, open_video

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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Image file or folder of images")
    source.add_argument("--video", help="Video file or folder of videos")
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
    add("--runs", type=int, help="Timed runs, median shown (default 5; 1 for a folder)")
    add("--stride", type=int, default=1, help="Classify every Nth video frame")
    add(
        "--overlay-size",
        type=float,
        default=0.0,
        help="Overlay size from 0 (compact, default) to 1 (twice as large)",
    )
    add(
        "--generate-dashboard",
        action="store_true",
        help="Draw a 1920x1080 dashboard instead of the overlay",
    )
    add("--output", help=f"Output file (single input) or folder (default {OUTPUT_DIR})")
    args = parser.parse_args(argv)
    tasks = [task for task, _ in args.model]
    if len(set(tasks)) != len(tasks):
        parser.error("give each task at most once")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if not 0 <= args.overlay_size <= 1:
        parser.error("--overlay-size must be between 0 and 1")
    return args


def load_predictors(args: argparse.Namespace) -> list[tuple[str, Predictor]]:
    """Load each checkpoint and check it matches its task."""
    loaded = []
    for task, checkpoint in args.model:
        predictor = Predictor(checkpoint, args.backend, args.device)
        check_task_classes(task, predictor.class_names)
        loaded.append((task, predictor))
    return loaded


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


def draw(
    args: argparse.Namespace, image: Image.Image, name: str, results: list[TaskResult]
):  # noqa: ANN201
    """Render the overlay, or the dashboard with ``--generate-dashboard``."""
    if args.generate_dashboard:
        return render_dashboard(image, name, results)
    return render_overlay(image, results, args.overlay_size)


def output_path(args: argparse.Namespace, path: Path, many: bool, ext: str) -> Path:
    """Where to save the result for ``path``."""
    if args.output and not many:
        return Path(args.output)
    suffix = "_dashboard" if args.generate_dashboard else ""
    return Path(args.output or OUTPUT_DIR) / f"{path.stem}{suffix}{ext}"


def run_images(args, paths, predictors, console, many) -> int:  # noqa: ANN001
    """Classify and draw each image; return the number that failed."""
    runs, failed = args.runs or (1 if many else 5), 0
    for i, path in enumerate(paths):
        output = output_path(args, path, many, ".jpg")
        try:
            image = load_image(path)
            with console.status(f"{path.name} ({i + 1}/{len(paths)})"):
                results = [
                    TaskResult(
                        task,
                        predictor.model_name,
                        predictor.backend,
                        median_prediction(predictor, image, runs, warmup=i == 0),
                    )
                    for task, predictor in predictors
                ]
            output.parent.mkdir(parents=True, exist_ok=True)
            draw(args, image, path.name, results).save(output, quality=92)
        except (OSError, ValueError) as err:
            failed += 1
            Console(stderr=True).print(f"[red]skipped[/red] {path.name}: {err}")
            continue
        print_results(console, path.name, results)
        console.print(f"saved [bold]{output}[/bold]")
    return failed


def run_video(args, path, predictors, console, many) -> None:  # noqa: ANN001
    """Classify every ``--stride``-th frame of one video and write an annotated mp4."""
    output, results = output_path(args, path, many, ".mp4"), []
    counts = {task: Counter() for task, _ in predictors}
    seconds = []
    with open_video(path) as (fps, total, frames), VideoWriter(output, fps) as writer:
        ticks = track(
            frames, description=path.name, total=total or None, console=console
        )
        for i, frame in enumerate(ticks):
            if i % args.stride == 0:
                results = [
                    TaskResult(task, p.model_name, p.backend, p.predict(frame))
                    for task, p in predictors
                ]
                for r in results:
                    counts[r.task][r.prediction.top1[0]] += 1
                seconds.append(sum(r.prediction.seconds for r in results))
            writer.write(draw(args, frame, path.name, results))
    if not seconds:
        raise ValueError(f"No frames could be read from {path}")
    table = Table(title=f"{path.name}: {len(seconds)} frames classified")
    for column in ("task", "frames per class"):
        table.add_column(column)
    for task, counter in counts.items():
        top = ", ".join(f"{k} {v}" for k, v in counter.most_common())
        table.add_row(task, top)
    console.print(table)
    mean = 1000 * sum(seconds) / len(seconds)
    console.print(f"saved [bold]{output}[/bold] (all models {mean:.1f} ms/frame)")


def main(argv: list[str] | None = None) -> int:
    """Run the demo; return a process exit code."""
    args = parse_args(argv)
    console, source = Console(), Path(args.image or args.video)
    try:
        if args.video:
            paths = list_media(source, VIDEO_EXTENSIONS, "videos")
        else:
            paths = list_media(source)
        predictors = load_predictors(args)
    except (FileNotFoundError, ValueError) as err:
        Console(stderr=True).print(f"[red]error:[/red] {err}")
        return 2

    many, failed = source.is_dir(), 0
    if args.image:
        failed = run_images(args, paths, predictors, console, many)
    else:
        for path in paths:
            try:
                run_video(args, path, predictors, console, many)
            except ValueError as err:
                failed += 1
                Console(stderr=True).print(f"[red]skipped[/red] {path.name}: {err}")
    if many:
        console.print(f"{len(paths) - failed}/{len(paths)} done")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
