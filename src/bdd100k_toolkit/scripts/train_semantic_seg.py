r"""
`bdd100k-train`: Hydra-driven semantic segmentation training.

Usage:
  bdd100k-train dataset=bdd100k-semantic-seg \
      model.architecture=Unet model.encoder_name=resnet34
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bdd100k_toolkit.segmentation.semantic.trainer import SemanticSegTrainer
from bdd100k_toolkit.utils.console import RichConsoleManager
from bdd100k_toolkit.utils.device import resolve_device

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@hydra.main(
    version_base=None, config_path=str(CONFIGS_DIR), config_name="config_semantic_seg"
)
def main(cfg: DictConfig) -> None:
    """Train a semantic segmentation model per the composed Hydra config."""
    console = RichConsoleManager.get_console()
    console.print(
        "\n[bold green]BDD100K-Toolkit Semantic Segmentation Training[/bold green]"
    )
    console.print(f"  Dataset: {cfg.dataset.name}")
    arch_line = f"  Architecture: {cfg.model.architecture} ({cfg.model.encoder_name})"
    console.print(arch_line)
    console.print(f"  Data dir: {cfg.dataset.data_dir}\n")

    trainer = SemanticSegTrainer(
        num_classes=cfg.dataset.num_classes,
        architecture=cfg.model.architecture,
        encoder_name=cfg.model.encoder_name,
        device=resolve_device(cfg.training.device),
    )
    result = trainer.train(
        data_dir=cfg.dataset.data_dir,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        imgsz=cfg.training.imgsz,
        output_dir=cfg.training.output_dir,
        workers=cfg.training.workers,
    )
    console.print(
        f"[bold green]Done.[/bold green] Model saved to: {result['model_path']}"
    )


if __name__ == "__main__":
    main()
