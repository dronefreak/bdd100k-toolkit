r"""
`bdd100k-train`: Hydra-driven classification training.

Usage:
  bdd100k-train dataset=bdd100k-weather model.name=yolo11n-cls
  bdd100k-train dataset=bdd100k-weather model.backend=timm model.name=convnext_tiny
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from bdd100k_toolkit.classification.backends import build_trainer
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
    console.print(f"  Model: {cfg.model.name} ({cfg.model.backend} backend)")
    console.print(f"  Data dir: {cfg.dataset.data_dir}\n")

    extra = OmegaConf.to_container(cfg.training.extra, resolve=True)
    if not isinstance(extra, dict):
        raise TypeError("training.extra must be a mapping of trainer arguments")

    trainer = build_trainer(
        cfg.model.backend, cfg.model.name, resolve_device(cfg.training.device)
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
        optimizer=cfg.training.optimizer,
        monitor=cfg.training.monitor,
        **{str(key): value for key, value in extra.items()},
    )
    console.print(
        f"[bold green]Done.[/bold green] Model saved to: {result['model_path']}"
    )


if __name__ == "__main__":
    main()
