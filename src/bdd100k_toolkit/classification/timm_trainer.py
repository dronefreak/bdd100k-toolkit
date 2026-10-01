"""
Classification training with ``timm`` models (optional ``timm`` extra).

``timm`` (PyTorch Image Models, by Hugging Face) offers over a thousand
architectures with pretrained weights behind one call,
``timm.create_model(name, pretrained=True, num_classes=n)``. This trainer is a
plain PyTorch loop around it, deliberately small, built for the BDD100K
attribute tasks rather than for generality:

- model selection and early stopping by a monitored metric on ``valid``, **macro
  F1** by default (not top-1), since the tasks are heavily imbalanced;
- an exponential moving average of the weights (``timm.utils.ModelEmaV3``, on by
  default); validation, ``best.pt`` and ``last.pt`` use the EMA weights;
- optional class balancing (``balance="loss"`` weights or ``"sampler"``);
- augmentations that keep the labels valid: no colour jitter by default
  (brightness *is* the time-of-day label), crops that keep the 16:9 aspect, and
  the same whole-image square resize at train and test time (no centre crop
  that would cut off the sky or the road edge);
- the same canonical ``train/valid/test/<class>/*.jpg`` folders as the
  Ultralytics backend.

Checkpoints are plain dicts (``backend="timm"``, ``model_name``,
``class_names``, ``imgsz``, ``mean``, ``std``, ``state_dict``) so they load with
``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from bdd100k_toolkit.utils.cls_metrics import (
    DEFAULT_MONITOR,
    check_monitor,
    classification_metrics,
)

BACKEND = "timm"
_BALANCE_MODES = ("none", "loss", "sampler")
_OPTIMIZERS = ("auto", "adamw", "sgd")
_DEFAULT_LR = 3e-4


def import_timm() -> Any:
    """Import timm, with an install hint when the optional extra is missing."""
    try:
        import timm
    except ImportError as err:
        raise ImportError(
            "timm is required for the 'timm' backend. "
            'Install with: pip install "bdd100k-toolkit[timm]"'
        ) from err
    return timm


def _torch_device(device: str) -> torch.device:
    """Accept ``"cpu"``, ``"cuda"``, ``"cuda:1"`` and a bare index like ``"0"``."""
    return torch.device(f"cuda:{device}" if device.isdigit() else device)


def _load_rgb(path: str, min_side: int) -> Image.Image:
    """
    Open a jpg as RGB, letting libjpeg decode at reduced size when possible.

    BDD100K images are 1280x720; decoding at half size is roughly twice as fast
    and still leaves at least ``min_side`` pixels on the short side, which is
    what dominates the data loading time of a small model.
    """
    with Image.open(path) as image:
        image.draft("RGB", (min_side, min_side))
        return image.convert("RGB")


def build_transforms(  # noqa: PLR0913
    imgsz: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    scale_min: float = 0.5,
    hflip: float = 0.5,
    color_jitter: float = 0.0,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train, eval)`` transforms; both squash the whole image to square."""
    train_steps: list[Any] = [
        # ratio ~ 16:9 so a crop resized to a square distorts like the full image
        transforms.RandomResizedCrop(
            imgsz, scale=(scale_min, 1.0), ratio=(1.5, 2.0), antialias=True
        ),
        transforms.RandomHorizontalFlip(hflip),
    ]
    if color_jitter > 0:
        train_steps.append(transforms.ColorJitter(color_jitter, color_jitter))
    tail = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    eval_steps = [transforms.Resize((imgsz, imgsz), antialias=True), *tail]
    return transforms.Compose([*train_steps, *tail]), transforms.Compose(eval_steps)


def make_dataset(
    split_dir: Path, transform: transforms.Compose, imgsz: int
) -> ImageFolder:
    """Build an ``ImageFolder`` over ``split_dir`` with the reduced-size jpg loader."""
    return ImageFolder(
        str(split_dir),
        transform=transform,
        loader=lambda path: _load_rgb(path, imgsz),
    )


def balance_weights(counts: np.ndarray, power: float) -> np.ndarray:
    """Per-class weights ``count ** -power``, scaled to mean 1 over the classes."""
    weights = np.power(np.maximum(counts, 1).astype(np.float64), -power)
    return weights / weights.mean()


class EvalResult(NamedTuple):
    """Outcome of one pass over a loader."""

    confusion: np.ndarray  # [true, predicted]
    top5: float  # top-k accuracy with k = min(5, num_classes)
    loss: float  # mean (unweighted) cross-entropy


@torch.no_grad()
def confusion_over_loader(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    num_classes: int,
    *,
    use_amp: bool = False,
) -> EvalResult:
    """Run ``model`` over ``loader`` and collect confusion matrix, top-k and loss."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    top_k = min(5, num_classes)
    hits = 0
    total = 0
    loss_sum = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        predicted = logits.argmax(dim=1).cpu().numpy()
        targets_np = targets.numpy()
        np.add.at(confusion, (targets_np, predicted), 1)
        top = logits.topk(top_k, dim=1).indices.cpu()
        hits += int((top == targets.unsqueeze(1)).any(dim=1).sum())
        loss_sum += float(
            nn.functional.cross_entropy(
                logits, targets.to(device), reduction="sum"
            ).item()
        )
        total += len(targets)
    return EvalResult(confusion, hits / max(total, 1), loss_sum / max(total, 1))


@dataclass
class _EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_balanced_accuracy: float
    val_macro_f1: float
    monitored: float
    lr: float
    seconds: float


class TimmClassificationTrainer:
    """Trains ``timm`` classifiers on the canonical ImageFolder layout."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        """
        Initialize TimmClassificationTrainer.

        Args:
            model_name: Any ``timm`` model name, e.g. ``"convnext_tiny"``,
                ``"resnet50"``, ``"efficientnet_b0"``. A specific pretrained
                tag can be given as ``"convnext_tiny.fb_in1k"``.
            device: ``"cuda"``, ``"cpu"``, ``"cuda:1"`` or a bare index.

        """
        self._timm = import_timm()
        self.model_name = model_name
        self.device = _torch_device(device)

    def train(  # noqa: PLR0913, PLR0917, PLR0915
        self,
        data_dir: str | Path,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float | None = None,
        imgsz: int = 224,
        output_dir: str | Path = "outputs",
        workers: int = 4,
        patience: int = 100,
        optimizer: str = "auto",
        monitor: str = DEFAULT_MONITOR,
        *,
        ema: bool = True,
        ema_decay: float = 0.999,
        grad_clip: float = 0.0,
        weight_decay: float = 0.05,
        label_smoothing: float = 0.0,
        balance: str = "none",
        balance_power: float = 0.5,
        warmup_epochs: int = 1,
        pretrained: bool = True,
        scale_min: float = 0.5,
        hflip: float = 0.5,
        color_jitter: float = 0.0,
        seed: int = 0,
        use_amp: bool = True,
    ) -> dict[str, Any]:
        """
        Train on ``data_dir/{train,valid}`` and keep the best ``monitor`` weights.

        Args:
            data_dir: Canonical root with ``train/`` and ``valid/``.
            epochs: Number of epochs.
            batch_size: Batch size.
            lr: Peak learning rate (default 3e-4, a fine-tuning value).
            imgsz: Square input size.
            output_dir: Run folder is ``output_dir/<model_name>/``.
            workers: DataLoader workers.
            patience: Epochs without improvement of ``monitor`` before stopping
                (0 disables early stopping).
            optimizer: ``"auto"``/``"adamw"`` or ``"sgd"``.
            monitor: Metric on ``valid`` that selects ``best.pt`` and drives
                early stopping: ``"macro_f1"`` (default),
                ``"balanced_accuracy"`` or ``"accuracy"``.
            ema: Keep an exponential moving average of the weights and use it
                for validation and for the saved checkpoints.
            ema_decay: Target EMA decay. It ramps up from 0 (``use_warmup``), so
                short runs are not dominated by the random-looking early weights.
            grad_clip: Clip the gradient norm to this value (0 disables).
            weight_decay: Optimizer weight decay.
            label_smoothing: Cross-entropy label smoothing.
            balance: ``"none"``, ``"loss"`` (class-weighted loss) or
                ``"sampler"`` (class-balanced sampling).
            balance_power: Weights are ``count ** -balance_power`` (1.0 is full
                inverse frequency, 0.5 is its square root).
            warmup_epochs: Linear LR warmup before the cosine decay.
            pretrained: Start from the pretrained weights.
            scale_min: Smallest crop area fraction for training.
            hflip: Horizontal flip probability.
            color_jitter: Brightness/contrast jitter (0 keeps time-of-day valid).
            seed: Random seed.
            use_amp: bfloat16 autocast on CUDA.

        Returns:
            dict with ``results`` (``monitor``, ``best_epoch``, ``best_value``,
            ``best_valid_macro_f1``), ``model_path`` (``best.pt``),
            ``output_dir`` and ``history``.

        """
        check_monitor(monitor)
        if balance not in _BALANCE_MODES:
            raise ValueError(
                f"balance must be one of {_BALANCE_MODES}, got {balance!r}"
            )
        if optimizer.lower() not in _OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {_OPTIMIZERS}, got {optimizer!r}"
            )
        _seed_everything(seed)
        data_dir = Path(data_dir)
        run_dir = Path(output_dir).resolve() / self.model_name
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)

        try:
            model = self._timm.create_model(
                self.model_name,
                pretrained=pretrained,
                num_classes=len(_class_names(data_dir / "train")),
            ).to(self.device)
        except Exception as err:
            raise ValueError(
                f"Failed to create model '{self.model_name}'. Browse the available "
                "timm models at "
                "https://github.com/huggingface/pytorch-image-models#models "
                f"(or call timm.list_models()): {err}"
            ) from err
        mean = tuple(model.pretrained_cfg.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(model.pretrained_cfg.get("std", (0.229, 0.224, 0.225)))
        train_tf, eval_tf = build_transforms(
            imgsz,
            mean,
            std,
            scale_min=scale_min,
            hflip=hflip,
            color_jitter=color_jitter,
        )
        train_ds = make_dataset(data_dir / "train", train_tf, imgsz)
        valid_ds = make_dataset(data_dir / "valid", eval_tf, imgsz)
        if train_ds.classes != valid_ds.classes:
            raise ValueError(
                f"train and valid have different classes: {train_ds.classes} vs "
                f"{valid_ds.classes}"
            )
        class_names = list(train_ds.classes)
        counts = np.bincount(train_ds.targets, minlength=len(class_names))
        weights = balance_weights(counts, balance_power)

        loss_weight = (
            torch.tensor(weights, dtype=torch.float32, device=self.device)
            if balance == "loss"
            else None
        )
        criterion = nn.CrossEntropyLoss(
            weight=loss_weight, label_smoothing=label_smoothing
        )
        sampler = (
            WeightedRandomSampler(
                weights[train_ds.targets].tolist(),
                num_samples=len(train_ds),
                replacement=True,
                generator=torch.Generator().manual_seed(seed),
            )
            if balance == "sampler"
            else None
        )
        pin = self.device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=workers,
            pin_memory=pin,
            drop_last=len(train_ds) > batch_size,
            persistent_workers=workers > 0,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=pin,
        )

        peak_lr = _DEFAULT_LR if lr is None else lr
        opt = _build_optimizer(optimizer, model, peak_lr, weight_decay)
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            _warmup_cosine(warmup_epochs * steps_per_epoch, epochs * steps_per_epoch),
        )
        amp = use_amp and self.device.type == "cuda"
        ema_model = (
            self._timm.utils.ModelEmaV3(model, decay=ema_decay, use_warmup=True)
            if ema
            else None
        )

        history: list[_EpochLog] = []
        best_value = -1.0
        best_epoch = 0
        best_macro_f1 = 0.0
        step = 0
        for epoch in range(1, epochs + 1):
            started = time.time()
            train_loss, step = self._train_one_epoch(
                model,
                train_loader,
                criterion,
                opt,
                scheduler,
                amp,
                ema_model,
                grad_clip,
                step,
            )
            # the EMA weights are what gets validated and saved
            eval_model = ema_model.module if ema_model is not None else model
            result = confusion_over_loader(
                eval_model, valid_loader, self.device, len(class_names), use_amp=amp
            )
            metrics = classification_metrics(result.confusion, class_names)
            value = float(metrics[monitor])
            log = _EpochLog(
                epoch,
                train_loss,
                result.loss,
                metrics["accuracy"],
                metrics["balanced_accuracy"],
                metrics["macro_f1"],
                value,
                opt.param_groups[0]["lr"],
                time.time() - started,
            )
            history.append(log)
            print(
                f"epoch {epoch}/{epochs}  train loss {log.train_loss:.4f}  "
                f"valid loss {log.val_loss:.4f}  acc {log.val_accuracy:.4f}  "
                f"macro-F1 {log.val_macro_f1:.4f}  ({log.seconds:.0f}s)",
                flush=True,
            )
            checkpoint = _checkpoint(
                eval_model,
                self.model_name,
                class_names,
                imgsz,
                mean,
                std,
                epoch,
                monitor,
                value,
                ema,
            )
            torch.save(checkpoint, run_dir / "weights" / "last.pt")
            if value > best_value:
                best_value, best_epoch = value, epoch
                best_macro_f1 = float(metrics["macro_f1"])
                torch.save(checkpoint, run_dir / "weights" / "best.pt")
            _write_history(run_dir / "results.csv", history, monitor)
            if patience > 0 and epoch - best_epoch >= patience:
                print(f"early stopping: no {monitor} gain since epoch {best_epoch}")
                break

        return {
            "results": {
                "monitor": monitor,
                "best_epoch": best_epoch,
                "best_value": best_value,
                "best_valid_macro_f1": best_macro_f1,
            },
            "model_path": str(run_dir / "weights" / "best.pt"),
            "output_dir": str(run_dir),
            "history": history,
        }

    def _train_one_epoch(  # noqa: PLR0913, PLR0917
        self,
        model: nn.Module,
        loader: DataLoader,  # type: ignore[type-arg]
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        amp: bool,
        ema_model: Any,
        grad_clip: float,
        step: int,
    ) -> tuple[float, int]:
        """Train one epoch; return ``(mean loss, updated global step)``."""
        model.train()
        total_loss = 0.0
        batches = 0
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=amp):
                loss = criterion(model(images), targets)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if ema_model is not None:
                ema_model.update(model, step=step)
            total_loss += float(loss.item())
            batches += 1
        return total_loss / max(batches, 1), step


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


def _class_names(train_dir: Path) -> list[str]:
    names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if not names:
        raise FileNotFoundError(f"No class folders found under {train_dir}")
    return names


def _build_optimizer(
    name: str, model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    if name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _warmup_cosine(warmup_steps: int, total_steps: int) -> Any:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def _checkpoint(  # noqa: PLR0913, PLR0917
    model: nn.Module,
    model_name: str,
    class_names: list[str],
    imgsz: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    epoch: int,
    monitor: str,
    value: float,
    ema: bool,
) -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "model_name": model_name,
        "class_names": class_names,
        "imgsz": imgsz,
        "mean": list(mean),
        "std": list(std),
        "epoch": epoch,
        "monitor": monitor,
        "monitor_value": value,
        "ema": ema,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


def _write_history(path: Path, history: list[_EpochLog], monitor: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "valid_loss",
                "valid_accuracy",
                "valid_balanced_accuracy",
                "valid_macro_f1",
                f"monitored_{monitor}",
                "lr",
                "seconds",
            ]
        )
        for log in history:
            writer.writerow(
                [
                    log.epoch,
                    f"{log.train_loss:.5f}",
                    f"{log.val_loss:.5f}",
                    f"{log.val_accuracy:.5f}",
                    f"{log.val_balanced_accuracy:.5f}",
                    f"{log.val_macro_f1:.5f}",
                    f"{log.monitored:.5f}",
                    f"{log.lr:.3e}",
                    f"{log.seconds:.1f}",
                ]
            )


def is_timm_checkpoint(path: str | Path) -> bool:
    """Return True for a checkpoint written by ``TimmClassificationTrainer``."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  # an Ultralytics .pt fails the safe load
        return False
    return isinstance(data, dict) and data.get("backend") == BACKEND


def evaluate_timm_checkpoint(  # noqa: PLR0913, PLR0917
    checkpoint_path: str | Path,
    data_dir: str | Path,
    split: str = "test",
    device: str = "cuda",
    batch_size: int = 128,
    workers: int = 4,
) -> tuple[np.ndarray, float, list[str]]:
    """
    Evaluate a timm checkpoint on ``data_dir/split``.

    Returns ``(confusion[true, pred], top5, class_names)``. Class order comes
    from the checkpoint and the split folder must contain the same classes.
    """
    timm = import_timm()
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_names: list[str] = data["class_names"]
    torch_device = _torch_device(device)
    model = timm.create_model(
        data["model_name"], pretrained=False, num_classes=len(class_names)
    )
    model.load_state_dict(data["state_dict"])
    model.to(torch_device)

    _, eval_tf = build_transforms(
        data["imgsz"], tuple(data["mean"]), tuple(data["std"])
    )
    dataset = make_dataset(Path(data_dir) / split, eval_tf, data["imgsz"])
    if dataset.classes != class_names:
        raise ValueError(
            f"{split} has classes {dataset.classes} but the checkpoint was trained "
            f"on {class_names}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch_device.type == "cuda",
    )
    result = confusion_over_loader(
        model, loader, torch_device, len(class_names), use_amp=False
    )
    return result.confusion, result.top5, class_names
