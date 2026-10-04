r"""
`bdd100k-evaluate --hf-model ID`: score a pretrained Hugging Face segmenter on BDD100K.

Zero-shot transfer: the Cityscapes checkpoints (SegFormer, Mask2Former, ...) share
BDD100K's 19 classes, so they can be scored without training. Each frame goes through
the checkpoint's own image processor with an explicit input size (its saved default is
a small square, far from the scale these models were trained at), and the prediction
is resized back to the label resolution with ``post_process_semantic_segmentation``.
The metrics follow the Cityscapes and official BDD100K evaluators: one dataset-wide
confusion matrix, ignored pixels dropped, native label resolution (see
``utils.miou.segmentation_report``).

Usage:
  bdd100k-evaluate --dataset bdd100k-semantic-seg \
      --hf-model nvidia/segformer-b2-finetuned-cityscapes-1024-1024 \
      --images-dir /path/to/seg/images/val --masks-dir /path/to/seg/labels/val
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rich.progress import track
from rich.table import Table

from bdd100k_toolkit.segmentation.semantic import get_spec, list_datasets
from bdd100k_toolkit.segmentation.semantic.datasets.bdd100k import (
    _pair_images_and_masks,
)
from bdd100k_toolkit.utils.console import RichConsoleManager
from bdd100k_toolkit.utils.device import resolve_device
from bdd100k_toolkit.utils.miou import confusion_matrix, segmentation_report

SIZE_MULTIPLE = 32
# Mask2Former never uses the Swin backbone's final norm (it has its own per-stage ones),
# and its Hub checkpoints omit it, so transformers reports it as missing.
BENIGN_MISSING = ("swin.layernorm.weight", "swin.layernorm.bias")
DTYPES = ("float32", "float16", "bfloat16")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained Hugging Face segmentation model on BDD100K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = parser.add_argument
    add("--hf-model", required=True, help="Hub id or local folder of the model")
    add("--revision", default="main", help="Hub branch, tag or commit to download")
    add("--dataset", required=True, choices=list_datasets(), help="Dataset key")
    add("--images-dir", required=True, help="Folder of <stem>.jpg images")
    add("--masks-dir", required=True, help="Folder of <stem>[_train_id].png masks")
    add("--short-side", type=int, default=1024, help="Input height (aspect kept)")
    add("--dtype", default="float32", choices=DTYPES, help="Inference precision")
    add("--device", default="auto", help="cuda / cpu / auto")
    add("--limit", type=int, help="Only evaluate the first N images (smoke test)")
    add("--output-dir", default="eval_outputs", help="Where metrics.json goes")
    return parser.parse_args(argv)


def input_size(image_size: tuple[int, int], short_side: int) -> dict[str, int]:
    """
    Processor size for an image: height ``short_side``, width scaled, rounded up to 32.

    ``image_size`` is PIL's ``(width, height)``. 1280x720 at 1024 gives 1024x1824.
    """
    width, height = image_size
    scaled = width * short_side / height
    return {
        "height": short_side,
        "width": math.ceil(scaled / SIZE_MULTIPLE) * SIZE_MULTIPLE,
    }


def load_model(
    name: str, device: str, dtype: str, revision: str = "main"
) -> tuple[Any, Any]:
    """Load ``(processor, model)``; Mask2Former is a universal-segmentation model."""
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForSemanticSegmentation,
        AutoModelForUniversalSegmentation,
    )

    try:
        model, info = AutoModelForSemanticSegmentation.from_pretrained(
            name, revision=revision, output_loading_info=True
        )
    except ValueError:
        model, info = AutoModelForUniversalSegmentation.from_pretrained(
            name, revision=revision, output_loading_info=True
        )
    missing = [k for k in info["missing_keys"] if not k.endswith(BENIGN_MISSING)]
    if missing:
        raise ValueError(
            f"{len(missing)} weights of {name} are missing from the checkpoint and "
            f"would be randomly initialised (e.g. {missing[0]}). Scores would be "
            "meaningless: likely a checkpoint/transformers version mismatch."
        )
    model = model.to(device=device, dtype=getattr(torch, dtype)).eval()
    return AutoImageProcessor.from_pretrained(name, revision=revision), model


def resolve_commit(name: str, revision: str) -> str | None:
    """Return the Hub commit hash ``revision`` points to (None if local or offline)."""
    from huggingface_hub import HfApi

    try:
        return HfApi().model_info(name, revision=revision).sha
    except (OSError, ValueError):
        return None


def check_classes(model: Any, classes: list[str]) -> None:
    """Raise unless the model's class names match ours in order."""
    names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    if names != classes:
        raise ValueError(
            f"Model classes do not match the dataset's {len(classes)} classes in "
            f"order.\n  model:   {names}\n  dataset: {classes}"
        )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Score the model on the paired images and masks; return the metrics report."""
    import torch

    spec = get_spec(args.dataset)
    classes = spec.classes
    device = resolve_device(args.device)
    processor, model = load_model(args.hf_model, device, args.dtype, args.revision)
    check_classes(model, classes)

    pairs = sorted(
        _pair_images_and_masks(Path(args.images_dir), Path(args.masks_dir)).values()
    )
    if args.limit:
        pairs = pairs[: args.limit]

    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for image_path, mask_path in track(pairs, description="Evaluating"):
        image = Image.open(image_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path))
        inputs = processor(
            images=image,
            size=input_size(image.size, args.short_side),
            return_tensors="pt",
        ).to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[mask.shape]
        )[0]
        confusion += confusion_matrix(
            prediction.cpu().numpy(), mask, len(classes), spec.ignore_index
        )

    report = segmentation_report(confusion, classes, spec.categories)
    report.update(
        model=args.hf_model,
        model_revision=resolve_commit(args.hf_model, args.revision),
        input_height=args.short_side,
        dtype=args.dtype,
        num_images=len(pairs),
        confusion_matrix=confusion.tolist(),
    )
    return report


def print_report(report: dict[str, Any]) -> None:
    """Print the summary and per-class tables."""
    console = RichConsoleManager.get_console()
    console.print(
        f"  mIoU: {report['miou']:.4f}   category mIoU: "
        f"{report.get('category_miou', float('nan')):.4f}\n"
        f"  fIoU: {report['fiou']:.4f}   pixel acc: {report['pacc']:.4f}   "
        f"images: {report['num_images']}"
    )
    table = Table(title="Per-class", header_style="bold cyan")
    table.add_column("Class")
    for column in ("IoU", "Precision", "Recall"):
        table.add_column(column, justify="right")
    for name, values in report["per_class"].items():
        table.add_row(
            name,
            f"{values['iou']:.3f}",
            f"{values['precision']:.3f}",
            f"{values['recall']:.3f}",
        )
    console.print(table)


def main(argv: list[str] | None = None) -> None:
    """Run the pretrained segmentation evaluation and save ``metrics.json``."""
    args = parse_args(argv)
    console = RichConsoleManager.get_console()
    console.print("\n[bold green]BDD100K-Toolkit Pretrained Segmenter[/bold green]")
    console.print(f"  Model: {args.hf_model}  (input height {args.short_side})")
    console.print(f"  Images: {args.images_dir}\n")
    report = evaluate(args)
    print_report(report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    console.print(f"\nMetrics saved to [bold]{output_dir / 'metrics.json'}[/bold]")


if __name__ == "__main__":
    main()
