"""
Semantic segmentation training via ``segmentation-models-pytorch``.

Unlike classification/detection, Ultralytics has no dense per-pixel
semantic segmentation trainer (its "segment" task is instance segmentation).
``segmentation-models-pytorch`` (SMP) is a small, actively maintained
library providing standard encoder/decoder architectures (Unet,
DeepLabV3+, ...) over a plain PyTorch training loop; this wrapper owns
just the training loop (data loading, optimizer, loss), delegating the
model architecture itself to SMP, the same "delegate, don't reimplement"
approach as ``UltralyticsClassificationTrainer``/``YOLODetectionTrainer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from bdd100k_toolkit.segmentation.semantic.base import IGNORE_INDEX


class ImageMaskDataset(Dataset):  # type: ignore[type-arg]
    """Loads ``images/*.jpg`` + ``masks/*.png`` pairs from a canonical split dir."""

    def __init__(self, split_dir: str | Path, imgsz: int = 512) -> None:
        """Initialize the dataset from a canonical ``images/``+``masks/`` split dir."""
        split_dir = Path(split_dir)
        self.image_paths = sorted((split_dir / "images").glob("*.jpg"))
        self.mask_dir = split_dir / "masks"
        self.imgsz = imgsz

    def __len__(self) -> int:
        """Return the number of image/mask pairs."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and resize one image/mask pair as tensors."""
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize((self.imgsz, self.imgsz), Image.Resampling.BILINEAR)
        )
        mask = Image.open(mask_path).resize(
            (self.imgsz, self.imgsz), Image.Resampling.NEAREST
        )

        image_arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_arr = np.asarray(mask, dtype=np.int64)
        return torch.from_numpy(image_arr), torch.from_numpy(mask_arr)


class SemanticSegTrainer:
    """Trains semantic segmentation models using ``segmentation-models-pytorch``."""

    def __init__(
        self,
        num_classes: int,
        architecture: str = "Unet",
        encoder_name: str = "resnet34",
        device: str = "cuda",
    ) -> None:
        """
        Initialize SemanticSegTrainer.

        Args:
            num_classes: Number of semantic classes (excluding "ignore").
            architecture: SMP model class name, e.g. 'Unet', 'DeepLabV3Plus'.
            encoder_name: SMP encoder backbone name, e.g. 'resnet34'.
            device: Device string ('cuda', 'cpu', ...).

        """
        try:
            import segmentation_models_pytorch as smp
        except ImportError as err:
            raise ImportError(
                "segmentation-models-pytorch is required for semantic "
                "segmentation training. Install with: "
                "pip install segmentation-models-pytorch"
            ) from err

        model_cls = getattr(smp, architecture)
        self.model = model_cls(
            encoder_name=encoder_name, classes=num_classes, activation=None
        )
        self.num_classes = num_classes
        self.device = device

    def train(  # noqa: PLR0913, PLR0917
        self,
        data_dir: str | Path,
        epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-3,
        imgsz: int = 512,
        output_dir: str | Path = "outputs",
        workers: int = 4,
    ) -> dict[str, Any]:
        """
        Train the model on the canonical ``{train,valid}/{images,masks}`` layout.

        Args:
            data_dir: Canonical segmentation root produced by an adapter's
                ``prepare_segmentation`` (must contain ``train/`` and
                ``valid/`` subfolders, each with ``images/``/``masks/``).
            epochs: Number of training epochs.
            batch_size: Batch size.
            lr: Learning rate (Adam).
            imgsz: Square input resolution images/masks are resized to.
            output_dir: Where to save the final model checkpoint.
            workers: Number of DataLoader workers.

        Returns:
            dict with keys: 'model_path', 'output_dir', 'final_train_loss'.

        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_ds = ImageMaskDataset(data_dir / "train", imgsz=imgsz)
        # Drop a trailing size-1 batch (whenever there's more than one full
        # batch) so BatchNorm never sees a single-sample batch during
        # training.
        drop_last = len(train_ds) > batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=drop_last,
        )

        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        final_loss = 0.0
        for _epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, masks)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
            final_loss = running_loss / max(1, len(train_loader))

        model_path = output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        return {
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "final_train_loss": final_loss,
        }
