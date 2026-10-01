"""The rich progress display: output, best marking, teardown, plain mode."""

from __future__ import annotations

import io
import re
import time

import pytest
from rich.console import Console

from bdd100k_toolkit.utils.training_progress import EpochRecord, TrainingProgress

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _console(*, terminal: bool) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, force_terminal=terminal, width=120), buffer


def _record(epoch: int, monitored: float, seconds: float = 3.0) -> EpochRecord:
    return EpochRecord(
        epoch=epoch,
        train_loss=1.0 / epoch,
        val_loss=0.9 / epoch,
        val_accuracy=0.7,
        val_balanced_accuracy=0.5,
        val_macro_f1=monitored,
        monitored=monitored,
        lr=3e-4,
        seconds=seconds,
    )


def _text(buffer: io.StringIO) -> str:
    return _ANSI.sub("", buffer.getvalue())


def test_end_epoch_reports_whether_the_epoch_is_the_best_so_far() -> None:
    console, _ = _console(terminal=False)
    with TrainingProgress(3, "macro_f1", enabled=True, console=console) as progress:
        results = []
        for epoch, value in enumerate([0.4, 0.6, 0.5], start=1):
            progress.start_epoch(epoch, 10, 4)
            results.append(progress.end_epoch(_record(epoch, value)))
    assert results == [True, True, False]


def test_epoch_lines_mark_the_best_and_show_the_metrics() -> None:
    console, buffer = _console(terminal=True)
    with TrainingProgress(2, "macro_f1", console=console) as progress:
        progress.start_epoch(1, 10, 4)
        progress.train_batch(10, 0.8, 3e-4)
        progress.start_valid()
        progress.valid_batch(4)
        progress.end_epoch(_record(1, 0.55))
        progress.start_epoch(2, 10, 4)
        progress.end_epoch(_record(2, 0.40))
    output = _text(buffer)
    lines = [line for line in output.splitlines() if line.startswith("epoch ")]
    assert "macro-F1 0.5500" in lines[0] and "best" in lines[0]
    assert "macro-F1 0.4000" in lines[1] and "best" not in lines[1]
    assert "best macro_f1 0.5500 (epoch 1)" in output  # overall bar status


def test_header_and_summary_table() -> None:
    console, buffer = _console(terminal=False)
    progress = TrainingProgress(2, "macro_f1", console=console)
    progress.header({"model": "timm resnet18", "ema": "on, decay 0.999"})
    progress.summary([_record(1, 0.4), _record(2, 0.6)], best_epoch=2)
    output = _text(buffer)
    assert "timm resnet18" in output and "on, decay 0.999" in output
    assert "best epoch 2 by macro_f1" in output
    assert "0.6000" in output


def test_disabled_mode_prints_plain_lines_without_live_output() -> None:
    console, buffer = _console(terminal=True)
    with TrainingProgress(2, "macro_f1", enabled=False, console=console) as progress:
        progress.start_epoch(1, 10, 4)
        progress.train_batch(10, 0.5, 1e-3)  # must be a harmless no-op
        progress.valid_batch(4)
        progress.end_epoch(_record(1, 0.5))
    assert buffer.getvalue() == ""  # plain mode uses print(), not the console


def test_disabled_mode_plain_line(capsys: pytest.CaptureFixture[str]) -> None:
    progress = TrainingProgress(5, "macro_f1", enabled=False)
    progress.end_epoch(_record(1, 0.5))
    progress.end_epoch(_record(2, 0.3))
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("epoch 1/5") and lines[0].endswith("* best")
    assert lines[1].startswith("epoch 2/5") and "best" not in lines[1]
    assert "\x1b" not in "".join(lines)  # no escape codes in log files


def test_live_display_is_torn_down_when_training_is_interrupted() -> None:
    console, _ = _console(terminal=True)
    progress = TrainingProgress(3, "macro_f1", console=console)
    with pytest.raises(KeyboardInterrupt), progress:
        progress.start_epoch(1, 10, 4)
        raise KeyboardInterrupt
    assert progress._progress is None  # Ctrl-C must not leave the terminal broken


def test_epoch_bar_has_no_throughput_but_image_bars_do() -> None:
    console, buffer = _console(terminal=True)
    with TrainingProgress(2, "macro_f1", console=console) as progress:
        progress.start_epoch(1, 1000, 100)
        progress.train_batch(250, 0.5, 1e-3)
        time.sleep(0.1)  # a rate needs two samples some time apart
        progress.train_batch(250, 0.5, 1e-3)
        assert progress._progress is not None
        progress._progress.refresh()
    output = _text(buffer)
    assert "img/s" in output  # from the train bar
    epoch_bar_frames = [
        line
        for line in output.splitlines()
        if line.lstrip(" ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏").startswith("epochs")
    ]
    assert epoch_bar_frames
    assert all("img/s" not in line for line in epoch_bar_frames)
