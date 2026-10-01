r"""
`bdd100k-prepare` entrypoint.

Converts a raw BDD100K download into the canonical per-class ImageFolder
splits.

Usage:
  bdd100k-prepare --dataset bdd100k-weather \
      --raw-dir /path/to/bdd100k_raw --output-dir /path/to/canonical_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bdd100k_toolkit.classification import get, list_datasets
from bdd100k_toolkit.utils.console import RichConsoleManager
from bdd100k_toolkit.utils.io import DEFAULT_JPEG_QUALITY, ImageOptions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a raw BDD100K download into canonical "
        "classification splits."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list_datasets(),
        help="Registered dataset key",
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Raw BDD100K download (images/100k + labels) or a Kaggle-style "
        "folder with {train,val}/<class>/*.jpg",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Folder with bdd100k_labels_images_{train,val}.json when it is not "
        "<raw-dir>/labels (official layout only)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        metavar="PX",
        help="Write train/valid images resized to at most PX wide (aspect kept, "
        "never upscaled) instead of hardlinking the 1280x720 originals. 512 makes "
        "Ultralytics training about 2.5-3.5x faster, with no accuracy loss seen in a "
        "6-epoch check. The test split keeps the originals unless --shrink-test.",
    )
    parser.add_argument(
        "--shrink-test",
        action="store_true",
        help="With --max-width, shrink the test split too",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality of resized images (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Processes used for resizing (default: all CPU cores)",
    )
    parser.add_argument(
        "--exclude-unknown",
        action="store_true",
        help="Drop the 'unknown' (official 'undefined') class; it is kept by default",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Canonical output root to write splits into"
    )
    return parser.parse_args()


def main() -> None:
    """Run the ``prepare-classification`` CLI."""
    args = parse_args()
    console = RichConsoleManager.get_console()
    adapter = get(args.dataset)
    console.print(f"[bold green]Preparing {adapter.spec.display_name}[/bold green]")
    console.print(f"  Classes ({adapter.spec.num_classes}): {adapter.spec.classes}")
    adapter.prepare_classification(
        Path(args.raw_dir),
        Path(args.output_dir),
        include_unknown=not args.exclude_unknown,
        labels_dir=Path(args.labels_dir) if args.labels_dir else None,
        image_options=ImageOptions(
            max_width=args.max_width,
            shrink_test=args.shrink_test,
            jpeg_quality=args.jpeg_quality,
            workers=args.workers,
        ),
    )
    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
