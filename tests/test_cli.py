"""Tests for the unified ``bdd100k-prepare/-train/-evaluate`` dispatch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bdd100k_toolkit import cli


def test_every_dataset_maps_to_one_task() -> None:
    tasks = cli.dataset_tasks()
    assert tasks["bdd100k-weather"] == "classification"
    assert tasks["bdd100k-period"] == "classification"
    assert tasks["bdd100k-scenario"] == "classification"
    assert tasks["bdd100k-detection"] == "detection"
    assert tasks["bdd100k-semantic-seg"] == "semantic-seg"


def test_unknown_dataset_lists_supported() -> None:
    with pytest.raises(KeyError, match="bdd100k-weather"):
        cli.task_for_dataset("nope")


@pytest.mark.parametrize(
    ("argv", "value", "rest"),
    [
        (["--task", "detection", "-x"], "detection", ["-x"]),
        (["--task=detection", "-x"], "detection", ["-x"]),
        (["-x"], None, ["-x"]),
    ],
)
def test_pop_option(argv: list[str], value: str | None, rest: list[str]) -> None:
    assert cli.pop_option(argv, "--task") == (value, rest)


def test_pop_option_requires_a_value() -> None:
    with pytest.raises(ValueError, match="needs a value"):
        cli.pop_option(["--task"], "--task")


def test_resolve_train_task_from_dataset_override() -> None:
    task, rest = cli.resolve_train_task(["dataset=bdd100k-detection", "a.b=1"])
    assert task == "detection"
    assert rest == ["dataset=bdd100k-detection", "a.b=1"]


def test_resolve_train_task_consumes_task_override() -> None:
    task, rest = cli.resolve_train_task(["task=semantic-seg", "model.encoder_name=x"])
    assert task == "semantic-seg"
    assert rest == ["model.encoder_name=x"]  # Hydra never sees ``task=``


def test_resolve_train_task_rejects_conflict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.resolve_train_task(["task=detection", "dataset=bdd100k-weather"])
    assert exc.value.code == 2
    assert "classification dataset" in capsys.readouterr().err


def test_train_without_task_or_dataset_exits_with_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["bdd100k-train", "training.epochs=1"])
    with pytest.raises(SystemExit) as exc:
        cli.train()
    assert exc.value.code == 2
    assert "bdd100k-detection" in capsys.readouterr().err


def test_prepare_help_lists_datasets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["bdd100k-prepare", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.prepare()
    assert exc.value.code == 0
    assert "bdd100k-semantic-seg" in capsys.readouterr().out


def test_prepare_without_arguments_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["bdd100k-prepare"])
    with pytest.raises(SystemExit) as exc:
        cli.prepare()
    assert exc.value.code == 2


def test_prepare_dispatches_classification(
    monkeypatch: pytest.MonkeyPatch, raw_bdd100k_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bdd100k-prepare",
            "--dataset",
            "bdd100k-weather",
            "--raw-dir",
            str(raw_bdd100k_dir),
            "--output-dir",
            str(out),
        ],
    )
    cli.prepare()
    assert (out / "test" / "foggy" / "v0.jpg").is_file()


def test_prepare_dispatches_detection_and_segmentation(
    monkeypatch: pytest.MonkeyPatch,
    raw_bdd100k_detection_dir: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "coco"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bdd100k-prepare",
            "--task=detection",
            "--dataset=bdd100k-detection",
            "--raw-dir",
            str(raw_bdd100k_detection_dir),
            "--output-dir",
            str(out),
        ],
    )
    with pytest.warns(UserWarning):
        cli.prepare()
    assert (out / "test" / "_annotations.coco.json").is_file()


def test_prepare_task_dataset_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["bdd100k-prepare", "--task", "detection", "--dataset", "bdd100k-weather"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.prepare()
    assert exc.value.code == 2
