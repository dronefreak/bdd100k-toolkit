"""The adapters also read per-image labels (labels/100k/<split>/**/<id>.json)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bdd100k_toolkit.classification import get as get_classification
from bdd100k_toolkit.detection import get as get_detection
from bdd100k_toolkit.utils.labels import label_source, read_entries


def _as_per_image(labels: Path, *, shard: bool = True) -> None:
    """Re-express ``bdd100k_labels_images_<split>.json`` as one JSON per image."""
    for split in ("train", "val"):
        path = labels / f"bdd100k_labels_images_{split}.json"
        for index, entry in enumerate(json.loads(path.read_text())):
            folder = (
                labels / "100k" / split / (f"shard_{index % 2:03d}" if shard else "")
            )
            folder.mkdir(parents=True, exist_ok=True)
            payload = {
                "name": entry["name"].removesuffix(".jpg"),
                "attributes": entry.get("attributes", {}),
                "frames": [{"objects": entry["labels"], "timestamp": 10000}],
            }
            (folder / f"{payload['name']}.json").write_text(json.dumps(payload))
        path.unlink()


def test_entries_round_trip_through_the_per_image_layout(
    raw_bdd100k_detection_dir: Path,
) -> None:
    labels = raw_bdd100k_detection_dir / "labels"
    before = read_entries(label_source(labels, "train"))  # type: ignore[arg-type]
    _as_per_image(labels)
    source = label_source(labels, "train")
    assert source is not None and source.is_dir()
    after = read_entries(source)
    assert [(e["name"], e["labels"]) for e in after] == sorted(
        [(e["name"], e["labels"]) for e in before]
    )
    assert label_source(labels, "test") is None


def test_a_missing_frame_is_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "100k" / "train"
    folder.mkdir(parents=True)
    (folder / "a.json").write_text(json.dumps({"name": "a", "frames": []}))
    with pytest.raises(ValueError, match="one frame"):
        read_entries(folder)


@pytest.mark.parametrize("shard", [False, True])
def test_detection_gives_the_same_coco_from_per_image_labels(
    raw_bdd100k_detection_dir: Path, tmp_path: Path, shard: bool
) -> None:
    raw = raw_bdd100k_detection_dir
    copy = tmp_path / "copy"
    shutil.copytree(raw, copy)
    legacy_out, per_image_out = tmp_path / "legacy", tmp_path / "per_image"
    with pytest.warns(UserWarning):
        get_detection("bdd100k-detection").prepare_coco(raw, legacy_out)
    _as_per_image(copy / "labels", shard=shard)
    with pytest.warns(UserWarning):
        get_detection("bdd100k-detection").prepare_coco(copy, per_image_out)
    for split in ("train", "valid"):
        name = "_annotations.coco.json"
        old = json.loads((legacy_out / split / name).read_text())
        new = json.loads((per_image_out / split / name).read_text())
        assert sorted(i["file_name"] for i in old["images"]) == sorted(
            i["file_name"] for i in new["images"]
        )
        assert len(old["annotations"]) == len(new["annotations"])


def test_classification_gives_the_same_folders_from_per_image_labels(
    raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(raw_bdd100k_dir, copy)
    legacy_out, per_image_out = tmp_path / "legacy", tmp_path / "per_image"
    get_classification("bdd100k-weather").prepare_classification(
        raw_bdd100k_dir, legacy_out
    )
    _as_per_image(copy / "labels")
    get_classification("bdd100k-weather").prepare_classification(copy, per_image_out)

    def listing(root: Path) -> set[tuple[str, ...]]:
        return {p.relative_to(root).parts for p in root.rglob("*.jpg")}

    assert listing(legacy_out) == listing(per_image_out)


def test_detection_reads_the_hugging_face_hub_layout(
    raw_bdd100k_detection_dir: Path, tmp_path: Path
) -> None:
    """
    ``dronefreak/BDD100K``: data/images_100k/100k + data/labels, sharded.

    The Hub copy also has a viewer ``metadata.jsonl`` next to the images; it is ignored.
    """
    raw = raw_bdd100k_detection_dir
    hub = tmp_path / "hub"
    shutil.copytree(raw / "images", hub / "data" / "images_100k")
    shutil.copytree(raw / "labels", hub / "data" / "labels")
    _as_per_image(hub / "data" / "labels")
    for split in (
        "train",
        "val",
    ):  # the Hub's viewer table next to the images is ignored
        (hub / "data" / "images_100k" / "100k" / split / "metadata.jsonl").write_text(
            '{"file_name": "x.jpg", "objects": {"bbox": [], "category": []}}\n'
        )
    legacy_out, hub_out = tmp_path / "legacy", tmp_path / "hub_out"
    with pytest.warns(UserWarning):
        get_detection("bdd100k-detection").prepare_coco(raw, legacy_out)
    with pytest.warns(UserWarning):
        get_detection("bdd100k-detection").prepare_coco(hub, hub_out)
    name = "_annotations.coco.json"
    for split in ("train", "valid"):
        old = json.loads((legacy_out / split / name).read_text())
        new = json.loads((hub_out / split / name).read_text())
        assert len(old["images"]) == len(new["images"])
        assert len(old["annotations"]) == len(new["annotations"])
