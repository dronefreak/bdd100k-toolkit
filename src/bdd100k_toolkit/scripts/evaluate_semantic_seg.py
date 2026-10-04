r"""
`bdd100k-evaluate`: evaluate a trained segmentation model.

Computes per-class IoU and mIoU on the canonical ``test`` split. With ``--hf-model``
it scores a pretrained Hugging Face model instead (see
``evaluate_semantic_seg_pretrained``).

Usage:
  bdd100k-evaluate --checkpoint <run>/model.pt \
      --dataset bdd100k-semantic-seg --data-dir /path/to/canonical_out \
      --architecture Unet --encoder-name resnet34
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from bdd100k_toolkit.cli import find_option
from bdd100k_toolkit.segmentation.semantic import get_spec, list_datasets
from bdd100k_toolkit.segmentation.semantic.trainer import ImageMaskDataset
from bdd100k_toolkit.utils.console import RichConsoleManager
from bdd100k_toolkit.utils.miou import confusion_matrix, mean_iou


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a BDD100K-Toolkit semantic segmentation model."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pt")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list_datasets(),
        help="Registered dataset key (determines class names/count)",
    )
    parser.add_argument(
        "--data-dir", required=True, help="Canonical segmentation root (has test/)"
    )
    parser.add_argument("--architecture", default="Unet")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument(
        "--split", default="test", help="Split to evaluate (default: test)"
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write metrics.json (default: alongside checkpoint)",
    )
    return parser.parse_args()


def main() -> None:
    """Run semantic segmentation evaluation and print/save per-class IoU + mIoU."""
    if find_option(sys.argv[1:], "--hf-model") is not None:
        from bdd100k_toolkit.scripts import evaluate_semantic_seg_pretrained

        evaluate_semantic_seg_pretrained.main()
        return
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader

    args = parse_args()
    console = RichConsoleManager.get_console()
    console.print(
        "\n[bold green]BDD100K-Toolkit Semantic Segmentation Evaluation[/bold green]"
    )
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Data dir: {args.data_dir}\n")

    spec = get_spec(args.dataset)
    model_cls = getattr(smp, args.architecture)
    model = model_cls(
        encoder_name=args.encoder_name, classes=spec.num_classes, activation=None
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    )
    model = model.to(args.device).eval()

    dataset = ImageMaskDataset(Path(args.data_dir) / args.split, imgsz=args.imgsz)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total_confusion = np.zeros((spec.num_classes, spec.num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(args.device))
            preds = logits.argmax(dim=1).cpu().numpy()
            for pred, mask in zip(preds, masks.numpy(), strict=True):
                total_confusion += confusion_matrix(
                    pred, mask, spec.num_classes, spec.ignore_index
                )

    metrics = mean_iou(total_confusion)
    per_class_iou = metrics["per_class_iou"]
    console.print(f"  mIoU: {metrics['miou']:.4f}")
    for cls_name, iou in zip(spec.classes, per_class_iou, strict=True):
        line = "n/a" if math.isnan(iou) else f"{iou:.4f}"
        console.print(f"    {cls_name}: {line}")

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "miou": metrics["miou"],
        "per_class_iou": dict(zip(spec.classes, per_class_iou, strict=True)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
