"""demo/classify.py end to end: one run per outcome that matters."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("timm")

from bdd100k_toolkit.classification.timm_trainer import (  # noqa: E402
    TimmClassificationTrainer,
)

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
TASKS = {
    "weather": {"clear": (230, 200, 40), "rainy": (40, 60, 200)},
    "period": {"daytime": (240, 220, 120), "night": (10, 10, 40)},
    "scenario": {"highway": (120, 120, 130), "tunnel": (20, 20, 25)},
}


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory, make_tiny_dataset) -> dict[str, Path]:  # noqa: ANN001
    """One tiny timm checkpoint per task, using that task's real class names."""
    paths = {}
    for task, classes in TASKS.items():
        root = tmp_path_factory.mktemp(task)
        data = make_tiny_dataset(root / "data", classes)
        result = TimmClassificationTrainer("resnet18", device="cpu").train(
            data, epochs=2, batch_size=6, lr=3e-3, imgsz=32, workers=0,
            pretrained=False, output_dir=root / "out", progress=False,
        )  # fmt: skip
        paths[task] = Path(result["model_path"])
    return paths


@pytest.fixture
def classify(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    return importlib.import_module("classify")


def _run(classify, image, out, models, *extra):  # noqa: ANN001, ANN202
    args = [
        "--image",
        str(image),
        "--device",
        "cpu",
        "--runs",
        "1",
        "--output",
        str(out),
    ]
    for task, checkpoint in models.items():
        args += ["--model", f"{task}={checkpoint}"]
    return classify.main([*args, *extra])


@pytest.fixture
def frame(tmp_path: Path) -> Path:
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (1280, 720), (40, 60, 200)).save(path)
    return path


def test_default_is_an_overlay_and_the_flag_makes_a_dashboard(
    classify,
    checkpoints,
    frame,
    tmp_path,  # noqa: ANN001
) -> None:
    overlay = tmp_path / "overlay.jpg"
    assert _run(classify, frame, overlay, checkpoints) == 0
    drawn = Image.open(overlay)
    assert drawn.size == (1280, 720)  # same size as the input frame
    assert drawn.tobytes() != Image.open(frame).tobytes()  # predictions were drawn

    dashboard = tmp_path / "dashboard.jpg"
    assert _run(classify, frame, dashboard, checkpoints, "--generate-dashboard") == 0
    assert Image.open(dashboard).size == (1920, 1080)

    one = tmp_path / "one.jpg"  # a single task works in both modes
    assert _run(classify, frame, one, {"period": checkpoints["period"]}) == 0
    assert Image.open(one).size == (1280, 720)


def test_unusual_frames_still_render(classify, checkpoints, tmp_path) -> None:  # noqa: ANN001
    for name, size in {"portrait": (720, 1280), "tiny": (33, 17)}.items():
        image = tmp_path / f"{name}.jpg"
        Image.new("RGB", size, (90, 90, 90)).save(image)
        assert (
            _run(
                classify,
                image,
                tmp_path / f"{name}_out.jpg",
                {"weather": checkpoints["weather"]},
            )
            == 0
        )


def test_a_checkpoint_under_the_wrong_task_is_refused(
    classify,
    checkpoints,
    frame,
    tmp_path,
    capsys,  # noqa: ANN001
) -> None:
    code = _run(classify, frame, tmp_path / "x.jpg", {"weather": checkpoints["period"]})
    assert code == 2
    assert "does not match task" in " ".join(capsys.readouterr().err.split())


def test_missing_image_and_duplicate_task_fail_cleanly(
    classify,
    checkpoints,
    frame,
    tmp_path,
    capsys,  # noqa: ANN001
) -> None:
    models = {"period": checkpoints["period"]}
    assert _run(classify, tmp_path / "nope.jpg", tmp_path / "x.jpg", models) == 2
    assert "Image not found" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        classify.main(
            ["--image", str(frame), "--model", "period=a.pt", "--model", "period=b.pt"]
        )
