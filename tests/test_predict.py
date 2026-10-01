"""Single-image inference: both backends, backend detection, task check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("timm")

from bdd100k_toolkit.classification.predict import (  # noqa: E402
    Prediction,
    Predictor,
    check_task_classes,
)
from bdd100k_toolkit.classification.timm_trainer import (  # noqa: E402
    TimmClassificationTrainer,
    evaluate_timm_checkpoint,
)

WEATHER = {"clear": (230, 200, 40), "rainy": (40, 60, 200), "snowy": (240, 240, 250)}


@pytest.fixture
def timm_checkpoint(tmp_path: Path, make_tiny_dataset) -> tuple[Path, Path]:  # noqa: ANN001
    """Return a tiny timm checkpoint trained on a weather subset, and its dataset."""
    data = make_tiny_dataset(tmp_path / "data", WEATHER)
    result = TimmClassificationTrainer("resnet18", device="cpu").train(
        data, epochs=3, batch_size=6, lr=3e-3, imgsz=32, workers=0,
        pretrained=False, output_dir=tmp_path / "out", progress=False,
    )  # fmt: skip
    return Path(result["model_path"]), data


def test_prediction_helpers() -> None:
    prediction = Prediction(["a", "b", "c"], [0.2, 0.7, 0.1], 0.004)
    assert prediction.ranked() == [("b", 0.7), ("a", 0.2), ("c", 0.1)]
    assert prediction.top1 == ("b", 0.7)
    assert prediction.margin == pytest.approx(0.5)


def test_timm_predictor_returns_a_probability_distribution(
    timm_checkpoint: tuple[Path, Path],
) -> None:
    checkpoint, data = timm_checkpoint
    predictor = Predictor(checkpoint, device="cpu")  # backend detected from the file
    assert (predictor.backend, predictor.model_name) == ("timm", "resnet18")
    assert predictor.class_names == ["clear", "rainy", "snowy"]
    prediction = predictor.predict(Image.open(data / "test" / "rainy" / "0.jpg"))
    assert sum(prediction.probabilities) == pytest.approx(1.0, abs=1e-5)
    assert prediction.top1[0] == "rainy" and prediction.seconds > 0


def test_predictor_matches_what_evaluation_reports(
    timm_checkpoint: tuple[Path, Path],
) -> None:
    """A prediction must agree with bdd100k-evaluate for the same images."""
    checkpoint, data = timm_checkpoint
    predictor = Predictor(checkpoint, device="cpu")
    confusion = np.zeros((3, 3), dtype=int)
    for true_index, name in enumerate(predictor.class_names):
        for jpg in sorted((data / "test" / name).glob("*.jpg")):
            guess = predictor.predict(Image.open(jpg)).top1[0]
            confusion[true_index, predictor.class_names.index(guess)] += 1
    evaluated, _, _ = evaluate_timm_checkpoint(
        checkpoint, data, device="cpu", workers=0
    )
    assert np.array_equal(confusion, evaluated)


def test_bad_inputs_are_clear_errors(
    timm_checkpoint: tuple[Path, Path], tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        Predictor(tmp_path / "nope.pt")
    with pytest.raises(ValueError, match="backend"):
        Predictor(timm_checkpoint[0], backend="onnx")


def test_task_check() -> None:
    check_task_classes("weather", ["clear", "foggy", "unknown"])
    check_task_classes("bdd100k-weather", ["clear", "rainy"])  # no `unknown`: fine
    with pytest.raises(ValueError, match="does not match task 'period'"):
        check_task_classes("period", ["clear", "foggy"])
    with pytest.raises(ValueError, match="Unknown task"):
        check_task_classes("lanes", ["clear"])


def test_ultralytics_predictor_with_the_real_library(
    tmp_path: Path,
    make_tiny_dataset,  # noqa: ANN001
) -> None:
    pytest.importorskip("ultralytics")
    from bdd100k_toolkit.classification.ultralytics_trainer import (
        UltralyticsClassificationTrainer,
    )

    data = make_tiny_dataset(tmp_path / "data", WEATHER)
    result = UltralyticsClassificationTrainer("yolo11n-cls", device="cpu").train(
        data, epochs=1, batch_size=6, imgsz=32, workers=0, pretrained=False,
        output_dir=tmp_path / "out", plots=False, amp=False, verbose=False,
    )  # fmt: skip
    predictor = Predictor(result["model_path"], device="cpu")
    assert (predictor.backend, predictor.model_name) == ("ultralytics", "yolo11n-cls")
    prediction = predictor.predict(Image.open(data / "test" / "clear" / "0.jpg"))
    assert sum(prediction.probabilities) == pytest.approx(1.0, abs=1e-4)
