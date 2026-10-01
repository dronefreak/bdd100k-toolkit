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
    )
    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
