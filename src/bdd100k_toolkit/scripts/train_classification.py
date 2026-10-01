r"""
`bdd100k-train`: Hydra-driven classification training.

Usage:
  bdd100k-train dataset=bdd100k-weather model.name=yolo11n-cls
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bdd100k_toolkit.classification.trainer import ClassificationTrainer
from bdd100k_toolkit.utils.console import RichConsoleManager
from bdd100k_toolkit.utils.device import resolve_device

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@hydra.main(
    version_base=None,
    config_path=str(CONFIGS_DIR),
    config_name="config_classification",
)
def main(cfg: DictConfig) -> None:
    """Train a classifier per the composed Hydra config."""
    console = RichConsoleManager.get_console()
    console.print("\n[bold green]BDD100K-Toolkit Classification Training[/bold green]")
    console.print(f"  Dataset: {cfg.dataset.name}")
    console.print(f"  Model: {cfg.model.name}")
    console.print(f"  Data dir: {cfg.dataset.data_dir}\n")

    trainer = ClassificationTrainer(
        model_name=cfg.model.name, device=resolve_device(cfg.training.device)
    )
    result = trainer.train(
        data_dir=cfg.dataset.data_dir,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        imgsz=cfg.training.imgsz,
        output_dir=cfg.training.output_dir,
        workers=cfg.training.workers,
        patience=cfg.training.patience,
    )
    console.print(
        f"[bold green]Done.[/bold green] Model saved to: {result['model_path']}"
    )


if __name__ == "__main__":
    main()
