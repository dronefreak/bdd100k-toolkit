#!/usr/bin/env python
r"""
Detect objects in images or videos with a trained BDD100K detector and draw the boxes.

  python demo/detect.py --image frame.jpg \
      --model experiments/bdd100k-detection/yolo26s/yolo26s/weights/best.pt

--image can be a file or a folder of images. --video takes a video file or a folder of
videos; every --stride-th frame is detected (the others reuse the last boxes) and an
annotated mp4 is written. Each picture gets boxes with class and score, a panel with the
class counts, and the inference time. Outputs go to demo/outputs/ (or --output: a file
for a single input, else a folder).
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

from bdd100k_toolkit.detection.predict import Detections, Detector
from bdd100k_toolkit.utils.draw_boxes import render_boxes
from bdd100k_toolkit.utils.images import list_media, load_image
from bdd100k_toolkit.utils.video import VIDEO_EXTENSIONS, VideoWriter, open_video

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add = parser.add_argument
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Image file or folder of images")
    source.add_argument("--video", help="Video file or folder of videos")
    add("--model", required=True, help="Detection checkpoint (best.pt)")
    add("--device", default="auto", help="cuda / cpu / auto")
    add("--conf", type=float, default=0.25, help="Minimum box score")
    add("--iou", type=float, default=0.7, help="NMS IoU threshold")
    add("--imgsz", type=int, help="Inference size (default: the training size)")
    add("--stride", type=int, default=1, help="Detect every Nth video frame")
    add("--output", help=f"Output file (single input) or folder (default {OUTPUT_DIR})")
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    return args


def output_path(args: argparse.Namespace, path: Path, many: bool, ext: str) -> Path:
    """Where to save the result for ``path``."""
    if args.output and not many:
        return Path(args.output)
    return Path(args.output or OUTPUT_DIR) / f"{path.stem}{ext}"


def run_images(args, paths, detector, console, many) -> int:  # noqa: ANN001
    """Detect and draw each image; return the number that failed."""
    failed = 0
    for i, path in enumerate(paths):
        output = output_path(args, path, many, ".jpg")
        try:
            image = load_image(path)
            with console.status(f"{path.name} ({i + 1}/{len(paths)})"):
                detections = detector.predict(image)
            output.parent.mkdir(parents=True, exist_ok=True)
            render_boxes(image, detections).save(output, quality=92)
        except (OSError, ValueError) as err:
            failed += 1
            Console(stderr=True).print(f"[red]skipped[/red] {path.name}: {err}")
            continue
        counts = ", ".join(f"{k} {v}" for k, v in detections.counts().most_common())
        console.print(
            f"{path.name}: {counts or 'no detections'} "
            f"({detections.seconds * 1000:.1f} ms) -> [bold]{output}[/bold]"
        )
    return failed


def run_video(args, path, detector, console, many) -> None:  # noqa: ANN001
    """Detect every ``--stride``-th frame of one video and write an annotated mp4."""
    output, detections = output_path(args, path, many, ".mp4"), None
    counts: Counter[str] = Counter()
    seconds: list[float] = []
    with open_video(path) as (fps, total, frames), VideoWriter(output, fps) as writer:
        ticks = track(
            frames, description=path.name, total=total or None, console=console
        )
        for i, frame in enumerate(ticks):
            if i % args.stride == 0:
                detections = detector.predict(frame)
                counts.update(detections.counts())
                seconds.append(detections.seconds)
            assert isinstance(detections, Detections)  # noqa: S101
            writer.write(render_boxes(frame, detections))
    if not seconds:
        raise ValueError(f"No frames could be read from {path}")
    table = Table(title=f"{path.name}: {len(seconds)} frames detected")
    for column in ("class", "boxes", "per frame"):
        table.add_column(column)
    for name, n in counts.most_common():
        table.add_row(name, str(n), f"{n / len(seconds):.1f}")
    console.print(table)
    mean = 1000 * sum(seconds) / len(seconds)
    console.print(f"saved [bold]{output}[/bold] ({mean:.1f} ms/frame)")


def main(argv: list[str] | None = None) -> int:
    """Run the demo; return a process exit code."""
    args = parse_args(argv)
    console, source = Console(), Path(args.image or args.video)
    try:
        if args.video:
            paths = list_media(source, VIDEO_EXTENSIONS, "videos")
        else:
            paths = list_media(source)
        detector = Detector(
            args.model, args.device, conf=args.conf, iou=args.iou, imgsz=args.imgsz
        )
    except (FileNotFoundError, ValueError) as err:
        Console(stderr=True).print(f"[red]error:[/red] {err}")
        return 2

    detector.predict(Image.new("RGB", (640, 360)))  # warm-up, so latencies are honest
    many, failed = source.is_dir(), 0
    if args.image:
        failed = run_images(args, paths, detector, console, many)
    else:
        for path in paths:
            try:
                run_video(args, path, detector, console, many)
            except ValueError as err:
                failed += 1
                Console(stderr=True).print(f"[red]skipped[/red] {path.name}: {err}")
    if many:
        console.print(f"{len(paths) - failed}/{len(paths)} done")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
