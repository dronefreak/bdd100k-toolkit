r"""
`bdd100k-train model.name=rfdetr-*`: Hydra-driven RF-DETR detection training.

Reached through ``train_detection.main`` when ``model.name`` is an RF-DETR model.
RF-DETR does not evaluate after training, so run ``bdd100k-evaluate`` on the
checkpoint afterwards.

Usage:
  bdd100k-train dataset=bdd100k-detection model.name=rfdetr-nano \
      dataset.dataset_dir=/path/to/coco_dataset
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from bdd100k_toolkit.detection.rfdetr import (
    build_model_kwargs,
    build_training_kwargs,
    load_model_class,
    normalize_model_name,
    seed_everything,
)
from bdd100k_toolkit.utils.console import RichConsoleManager
from bdd100k_toolkit.utils.device import resolve_device

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@hydra.main(
    version_base=None,
    config_path=str(CONFIGS_DIR),
    config_name="config_detection_rfdetr",
)
def train(cfg: DictConfig) -> None:
    """Train the configured RF-DETR model."""
    seed_everything(int(cfg.training.seed))
    console = RichConsoleManager.get_console()
    console.print(OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.training.device)
    model_class = load_model_class(str(cfg.model.name))
    model = model_class(**build_model_kwargs(cfg, device))
    training_kwargs = build_training_kwargs(cfg, device)

    console.print(
        f"[bold cyan]Training {normalize_model_name(str(cfg.model.name))}"
        f"[/bold cyan]\n  Dataset: {training_kwargs['dataset_dir']}"
        f"\n  Output: {training_kwargs['output_dir']}"
    )
    model.train(**training_kwargs)
    console.print(
        f"[bold green]Done.[/bold green] Evaluate with: bdd100k-evaluate "
        f"--dataset {cfg.dataset.name} --model {cfg.model.name} --checkpoint "
        f"{training_kwargs['output_dir']}/checkpoint_best_total.pth "
        f"--dataset-dir {training_kwargs['dataset_dir']}"
    )


def main() -> None:
    """Run the RF-DETR training CLI entrypoint."""
    train()


if __name__ == "__main__":
    main()
