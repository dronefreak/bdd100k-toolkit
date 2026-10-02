r"""
`bdd100k-train`: Hydra-driven YOLO detection training.

Trains the configured YOLO model then evaluates the best checkpoint,
mirroring DetectionBench's ``scripts/train_yolo.py`` (trimmed to this
project's YOLO-only scope, with no RF-DETR dispatch).

Usage:
  bdd100k-train dataset=bdd100k-detection model.name=yolo11n
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from bdd100k_toolkit.detection import get_spec
from bdd100k_toolkit.detection.trainer import YOLODetectionTrainer
from bdd100k_toolkit.scripts.evaluate_detection import (
    EvaluationOptions,
    evaluate_detection,
    print_metrics_table,
)
from bdd100k_toolkit.utils.console import RichConsoleManager
from bdd100k_toolkit.utils.device import resolve_device

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@hydra.main(
    version_base=None, config_path=str(CONFIGS_DIR), config_name="config_detection"
)
def train_and_evaluate(cfg: DictConfig) -> None:
    """Train the configured YOLO detector and evaluate the best checkpoint."""
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg))
    trainer = YOLODetectionTrainer(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        device=resolve_device(cfg.training.device),
    )

    results = trainer.train(
        dataset_yaml=cfg.training.dataset_yaml,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        imgsz=cfg.training.imgsz,
        output_dir=cfg.training.output_dir,
        workers=cfg.training.workers,
        patience=cfg.training.patience,
        optimizer=cfg.training.optimizer,
        cos_lr=cfg.training.cos_lr,
        use_amp=cfg.training.use_amp,
        **dict(OmegaConf.to_container(cfg.training.extra, resolve=True)),  # type: ignore[arg-type]
    )
    if not results["model_path"]:
        raise RuntimeError("Training completed without producing a model checkpoint.")

    console.print(f"  Best model saved to: {results['model_path']}")
    console.print(f"  All artifacts saved to: {results['output_dir']}")

    console.print(
        f"\n[bold cyan]Evaluating the trained model on the "
        f"'{cfg.evaluation.split}' split...[/bold cyan]"
    )
    spec = get_spec(cfg.dataset.name)
    metrics = evaluate_detection(
        EvaluationOptions(
            checkpoint_path=results["model_path"],
            dataset_yaml=cfg.evaluation.dataset_yaml,
            class_names=spec.classes,
            num_classes=cfg.evaluation.num_classes,
            device=resolve_device(cfg.evaluation.device),
            output_dir=Path(cfg.evaluation.output_dir),
            save_predictions=cfg.evaluation.save_predictions,
            split=cfg.evaluation.split,
        )
    )
    console.print("\n[bold green]Evaluation metrics:[/bold green]")
    print_metrics_table(cfg.model.name, metrics)

    metrics_path = Path(cfg.evaluation.output_dir) / "metrics.json"
    serializable: dict[str, Any] = {
        k: v for k, v in metrics.items() if k != "per_class"
    }
    if "per_class" in metrics:
        serializable["per_class"] = metrics["per_class"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(serializable, indent=2))
    console.print(f"\n✓ Metrics saved to [bold]{metrics_path}[/bold]")


def main() -> None:
    """Run the training CLI entrypoint."""
    train_and_evaluate()


if __name__ == "__main__":
    main()
