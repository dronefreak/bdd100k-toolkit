r"""
`bdd100k-prepare` entrypoint.

Converts a raw BDD100K 10K-images download into the canonical
``{train,valid,test}/{images,masks}`` semantic segmentation layout.

Usage:
  bdd100k-prepare --dataset bdd100k-semantic-seg \
      --raw-dir /path/to/bdd100k_raw --output-dir /path/to/canonical_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bdd100k_toolkit.segmentation.semantic import get, list_datasets
from bdd100k_toolkit.utils.console import RichConsoleManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a raw BDD100K download into canonical semantic "
        "segmentation splits."
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
        help="Raw BDD100K download (images/10k + labels/sem_seg/masks)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Canonical output root to write splits into"
    )
    return parser.parse_args()


def main() -> None:
    """Run the ``prepare-semantic-seg`` CLI."""
    args = parse_args()
    console = RichConsoleManager.get_console()
    adapter = get(args.dataset)
    console.print(f"[bold green]Preparing {adapter.spec.display_name}[/bold green]")
    console.print(f"  Classes ({adapter.spec.num_classes}): {adapter.spec.classes}")
    adapter.prepare_segmentation(Path(args.raw_dir), Path(args.output_dir))
    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
