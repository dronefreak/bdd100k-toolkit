"""timm backend: training loop, checkpoints, balancing, evaluation (CPU, tiny)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("timm")

import torch  # noqa: E402

from bdd100k_toolkit.classification.backends import build_trainer  # noqa: E402
from bdd100k_toolkit.classification.timm_trainer import (  # noqa: E402
    TimmClassificationTrainer,
    _load_rgb,
    balance_weights,
    build_transforms,
    evaluate_timm_checkpoint,
    is_timm_checkpoint,
)


@pytest.fixture(autouse=True)
def few_threads() -> object:
    """Tiny CPU models run faster, and don't thrash a busy machine, on 2 threads."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


_COLORS = {"red": (220, 20, 20), "green": (20, 200, 20), "blue": (20, 20, 220)}


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Three classes of flat colour images in canonical train/valid/test folders."""
    sizes = {"train": 12, "valid": 4, "test": 4}
    for split, count in sizes.items():
        for name, color in _COLORS.items():
            folder = tmp_path / split / name
            folder.mkdir(parents=True)
            for i in range(count):
                jitter = tuple(min(255, c + i) for c in color)
                Image.new("RGB", (64, 36), jitter).save(folder / f"{i}.jpg")
    return tmp_path


def _train(data: Path, out: Path, **kwargs: object) -> dict:
    trainer = TimmClassificationTrainer("resnet18", device="cpu")
    params: dict = {
        "epochs": 2,
        "batch_size": 6,
        "imgsz": 32,
        "workers": 0,
        "pretrained": False,
        "output_dir": out,
    }
    params.update(kwargs)
    return trainer.train(data, **params)


def test_trains_and_writes_checkpoints_and_history(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    result = _train(tiny_dataset, tmp_path / "out")
    run = Path(result["output_dir"])
    assert (run / "weights" / "best.pt").is_file()
    assert (run / "weights" / "last.pt").is_file()
    rows = list(csv.DictReader((run / "results.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert result["results"]["best_valid_macro_f1"] >= 0.0

    checkpoint = torch.load(result["model_path"], weights_only=True)
    assert checkpoint["backend"] == "timm"
    assert checkpoint["model_name"] == "resnet18"
    assert checkpoint["class_names"] == ["blue", "green", "red"]  # sorted folders
    assert checkpoint["imgsz"] == 32


def test_it_learns_a_trivially_separable_task(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    result = _train(tiny_dataset, tmp_path / "out", epochs=8, lr=3e-3, batch_size=4)
    assert result["results"]["best_valid_macro_f1"] > 0.9


@pytest.mark.parametrize("balance", ["loss", "sampler"])
def test_balance_modes_run(tiny_dataset: Path, tmp_path: Path, balance: str) -> None:
    result = _train(tiny_dataset, tmp_path / "out", epochs=1, balance=balance)
    assert Path(result["model_path"]).is_file()


def test_invalid_options_are_rejected(tiny_dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="balance"):
        _train(tiny_dataset, tmp_path / "o", balance="magic")
    with pytest.raises(ValueError, match="optimizer"):
        _train(tiny_dataset, tmp_path / "o", optimizer="rmsprop")
    with pytest.raises(TypeError):  # typos in training.extra fail loudly
        _train(tiny_dataset, tmp_path / "o", weight_decayy=0.1)


def test_early_stopping_by_macro_f1(tiny_dataset: Path, tmp_path: Path) -> None:
    result = _train(tiny_dataset, tmp_path / "o", epochs=30, lr=1e-9, patience=2)
    assert len(result["history"]) < 30


def test_checkpoint_detection_and_evaluation(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    result = _train(tiny_dataset, tmp_path / "out", epochs=3, lr=3e-3)
    assert is_timm_checkpoint(result["model_path"])
    other = tmp_path / "other.pt"
    torch.save({"model": "not timm"}, other)
    assert not is_timm_checkpoint(other)
    assert not is_timm_checkpoint(tmp_path / "missing.pt")

    confusion, top5, names = evaluate_timm_checkpoint(
        result["model_path"], tiny_dataset, split="test", device="cpu", workers=0
    )
    assert names == ["blue", "green", "red"]
    assert confusion.shape == (3, 3)
    assert confusion.sum() == 12
    assert confusion.sum(axis=1).tolist() == [4, 4, 4]  # rows are true classes
    assert top5 == 1.0  # k = min(5, 3) classes


def test_evaluation_rejects_mismatched_classes(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    result = _train(tiny_dataset, tmp_path / "out", epochs=1)
    (tiny_dataset / "test" / "red").rename(tiny_dataset / "test" / "purple")
    with pytest.raises(ValueError, match="trained on"):
        evaluate_timm_checkpoint(
            result["model_path"], tiny_dataset, device="cpu", workers=0
        )


def test_balance_weights_math() -> None:
    counts = np.array([100, 10, 1])
    full = balance_weights(counts, 1.0)
    assert full.mean() == pytest.approx(1.0)
    assert full[2] / full[0] == pytest.approx(100.0)
    root = balance_weights(counts, 0.5)
    assert root[2] / root[0] == pytest.approx(10.0)
    assert balance_weights(counts, 0.0).tolist() == [1.0, 1.0, 1.0]
    assert np.isfinite(balance_weights(np.array([5, 0]), 1.0)).all()


def test_transforms_keep_labels_valid() -> None:
    train_tf, eval_tf = build_transforms(32, (0.5,) * 3, (0.5,) * 3)
    image = Image.new("RGB", (160, 90), (200, 30, 30))
    assert tuple(train_tf(image).shape) == (3, 32, 32)
    assert tuple(eval_tf(image).shape) == (3, 32, 32)
    names = [type(t).__name__ for t in train_tf.transforms]
    assert "ColorJitter" not in names  # brightness is the time-of-day label
    assert "CenterCrop" not in [type(t).__name__ for t in eval_tf.transforms]
    assert torch.equal(eval_tf(image), eval_tf(image))  # deterministic


def test_reduced_size_decode_keeps_enough_pixels(tmp_path: Path) -> None:
    path = tmp_path / "big.jpg"
    Image.new("RGB", (1280, 720), (10, 20, 30)).save(path)
    image = _load_rgb(str(path), 224)
    assert image.mode == "RGB"
    assert min(image.size) >= 224
    assert image.size[0] / image.size[1] == pytest.approx(16 / 9, rel=0.02)


def test_factory_builds_both_backends_and_rejects_unknown() -> None:
    assert build_trainer("timm", "resnet18", "cpu").__class__.__name__ == (
        "TimmClassificationTrainer"
    )
    with pytest.raises(ValueError, match="Unknown backend"):
        build_trainer("nope", "x", "cpu")


def test_missing_timm_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "timm", None)  # makes `import timm` fail
    with pytest.raises(ImportError, match=r"bdd100k-toolkit\[timm\]"):
        TimmClassificationTrainer("resnet18")


def test_unknown_timm_model_gives_a_helpful_error_with_the_cause(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    trainer = TimmClassificationTrainer("not_a_real_model", device="cpu")
    with pytest.raises(ValueError, match="not_a_real_model") as info:
        trainer.train(tiny_dataset, output_dir=tmp_path / "o", workers=0)
    assert "pytorch-image-models" in str(info.value)
    assert info.value.__cause__ is not None  # the original error is chained
