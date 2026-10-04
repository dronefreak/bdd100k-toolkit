"""Zero-shot Hugging Face segmenter evaluation: metrics, sizing and an offline run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from bdd100k_toolkit.scripts.evaluate_semantic_seg_pretrained import (
    check_classes,
    input_size,
    main,
)
from bdd100k_toolkit.segmentation.semantic import get_spec
from bdd100k_toolkit.utils.miou import category_confusion, segmentation_report
from PIL import Image

CLASSES = ["a", "b", "c"]


def test_report_matches_hand_computation() -> None:
    confusion = np.array([[8, 2, 0], [1, 5, 0], [0, 0, 0]])
    report = segmentation_report(confusion, CLASSES)
    # class a: 8 / (10 + 9 - 8); class b: 5 / (6 + 7 - 5); class c is absent
    assert report["per_class"]["a"]["iou"] == pytest.approx(8 / 11)
    assert report["per_class"]["b"]["iou"] == pytest.approx(5 / 8)
    assert report["miou"] == pytest.approx((8 / 11 + 5 / 8) / 2)  # c has no GT
    assert report["pacc"] == pytest.approx(13 / 16)
    assert report["fiou"] == pytest.approx((8 / 11 * 10 + 5 / 8 * 6) / 16)
    assert report["per_class"]["a"]["precision"] == pytest.approx(8 / 9)
    assert report["per_class"]["a"]["recall"] == pytest.approx(0.8)


def test_predicting_an_absent_class_does_not_change_miou() -> None:
    # Official BDD100K: only classes present in the ground truth are averaged.
    confusion = np.array([[5, 0, 5], [0, 4, 0], [0, 0, 0]])
    assert segmentation_report(confusion, CLASSES)["miou"] == pytest.approx(
        (5 / 10 + 1) / 2
    )


def test_category_iou_counts_within_category_confusion_as_correct() -> None:
    confusion = np.array([[6, 4, 0], [0, 10, 0], [0, 0, 5]])
    categories = {"ab": ["a", "b"], "c": ["c"]}
    assert category_confusion(confusion, CLASSES, categories).tolist() == [
        [20, 0],
        [0, 5],
    ]
    report = segmentation_report(confusion, CLASSES, categories)
    assert report["category_miou"] == 1.0


def test_input_size_keeps_aspect_and_rounds_to_32() -> None:
    assert input_size((1280, 720), 1024) == {"height": 1024, "width": 1824}
    assert input_size((2048, 1024), 1024) == {"height": 1024, "width": 2048}


def test_bdd_categories_cover_every_class_once() -> None:
    spec = get_spec("bdd100k-semantic-seg")
    assert spec.categories is not None
    members = [name for group in spec.categories.values() for name in group]
    assert sorted(members) == sorted(spec.classes)


def test_class_mismatch_is_rejected() -> None:
    model = type("M", (), {"config": type("C", (), {"id2label": {0: "x"}})()})()
    with pytest.raises(ValueError, match="do not match"):
        check_classes(model, ["road"])


def test_cli_scores_a_tiny_local_segformer(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    from transformers import SegformerImageProcessor

    spec = get_spec("bdd100k-semantic-seg")
    config = SegformerConfig(
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 1, 2, 2],
        decoder_hidden_size=16,
        id2label=dict(enumerate(spec.classes)),
        label2id={name: i for i, name in enumerate(spec.classes)},
    )
    model_dir = tmp_path / "tiny-segformer"
    SegformerForSemanticSegmentation(config).save_pretrained(model_dir)
    SegformerImageProcessor().save_pretrained(model_dir)

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    rng = np.random.default_rng(0)
    for stem in ("a", "b"):
        Image.fromarray(rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)).save(
            images / f"{stem}.jpg"
        )
        mask = rng.integers(0, spec.num_classes, (72, 128), dtype=np.uint8)
        mask[:8] = spec.ignore_index
        Image.fromarray(mask).save(masks / f"{stem}_train_id.png")

    out = tmp_path / "out"
    main(
        [
            "--dataset", "bdd100k-semantic-seg", "--hf-model", str(model_dir),
            "--images-dir", str(images), "--masks-dir", str(masks),
            "--short-side", "64", "--device", "cpu", "--output-dir", str(out),
        ]
    )  # fmt: skip
    report = json.loads((out / "metrics.json").read_text())
    assert report["num_images"] == 2
    counted = sum(map(sum, report["confusion_matrix"]))
    assert counted == 2 * (72 - 8) * 128  # ignored rows are dropped, nothing else
    assert 0.0 <= report["miou"] <= 1.0
    assert set(report["category_iou"]) == set(spec.categories or {})


def test_missing_checkpoint_weights_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    import transformers
    from bdd100k_toolkit.scripts.evaluate_semantic_seg_pretrained import load_model

    def fake(name: str, **kwargs: object):
        info = {
            "missing_keys": ["encoder.block.0.q_proj.weight", "swin.layernorm.bias"]
        }
        return object(), info

    monkeypatch.setattr(
        transformers.AutoModelForSemanticSegmentation, "from_pretrained", fake
    )
    with pytest.raises(ValueError, match="1 weights .* missing"):
        load_model("some/model", "cpu", "float32")
