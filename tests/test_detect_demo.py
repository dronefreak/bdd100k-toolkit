"""demo/detect.py end to end with an untrained tiny detector, and the box renderer."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("ultralytics")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from bdd100k_toolkit.detection.predict import Detections  # noqa: E402
from bdd100k_toolkit.utils.draw_boxes import render_boxes  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from ultralytics import YOLO

    path = tmp_path_factory.mktemp("run") / "tiny" / "weights" / "best.pt"
    path.parent.mkdir(parents=True)
    YOLO("yolo11n.yaml").save(str(path))
    return path


def test_render_keeps_size_and_draws_something() -> None:
    image = Image.new("RGB", (640, 360), (90, 90, 90))
    boxes = [(100.0, 100.0, 300.0, 250.0), (400.0, 50.0, 420.0, 70.0)]
    detections = Detections(boxes, [0.9, 0.4], [0, 1], ["car", "person"], 0.01)
    out = render_boxes(image, detections)
    assert out.size == image.size
    assert out.getpixel((100, 150)) != (90, 90, 90)  # box edge was painted
    empty = render_boxes(image, Detections([], [], [], ["car"], 0.01))
    assert empty.size == image.size


def test_image_folder_and_video(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(DEMO_DIR))
    detect = importlib.import_module("detect")
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (320, 180), (120, 120, 120)).save(frames / "a.png")
    (frames / "broken.jpg").write_bytes(b"not an image")
    out = tmp_path / "out"
    args = ["--device", "cpu", "--conf", "0.0", "--imgsz", "64"]
    args += ["--model", str(checkpoint), "--output", str(out)]
    assert detect.main(["--image", str(frames), *args]) == 1
    assert [p.name for p in out.iterdir()] == ["a.jpg"]

    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter.fourcc(*"mp4v"), 10, (160, 90))
    for _ in range(6):
        writer.write(np.full((90, 160, 3), 128, np.uint8))
    writer.release()
    video_out = tmp_path / "video_out.mp4"
    assert (
        detect.main(
            [
                "--video",
                str(clip),
                "--stride",
                "2",
                *args[:-2],
                "--output",
                str(video_out),
            ]
        )
        == 0
    )
    assert int(cv2.VideoCapture(str(video_out)).get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    assert detect.main(["--image", str(tmp_path / "missing"), *args]) == 2
