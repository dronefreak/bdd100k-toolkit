"""
Rich progress display for the hand-written training loops.

Ultralytics already renders its own progress bars; the PyTorch loop of the
``timm`` backend uses this instead. It shows, live:

- a header with the run settings (model, parameters, device, EMA, monitor...);
- the current epoch's train and valid bars, measured in images so the rate is
  images per second, with the running loss and learning rate;
- an overall epochs bar with the best monitored value so far;

and prints one result line per finished epoch (the best epoch is marked) plus a
summary table at the end. With ``enabled=False`` it prints one plain line per
epoch and no live display, which suits log files and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from bdd100k_toolkit.utils.console import RichConsoleManager


@dataclass
class EpochRecord:
    """Metrics of one finished epoch (all validation values are on ``valid``)."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_balanced_accuracy: float
    val_macro_f1: float
    monitored: float
    lr: float
    seconds: float


class _ThroughputColumn(ProgressColumn):
    """Images per second (the tasks advance by images, not batches)."""

    def render(self, task: Any) -> Text:
        if not task.fields.get("images"):  # the overall epochs bar has no rate
            return Text("")
        speed = task.finished_speed or task.speed
        return Text(f"{speed:,.0f} img/s", style="cyan") if speed else Text("")


class TrainingProgress:
    """Live progress, per-epoch lines and a final table for a training run."""

    _LOSS_SMOOTHING = 0.9

    def __init__(
        self,
        epochs: int,
        monitor: str,
        *,
        enabled: bool = True,
        console: Console | None = None,
    ) -> None:
        """
        Initialize TrainingProgress.

        Args:
            epochs: Planned number of epochs (early stopping may end sooner).
            monitor: Name of the metric that selects the best epoch.
            enabled: Draw live bars; otherwise only plain per-epoch lines.
            console: Rich console to use (the shared one by default).

        """
        self.epochs = epochs
        self.monitor = monitor
        self.enabled = enabled
        self.console = console or RichConsoleManager.get_console()
        self._progress: Progress | None = None
        self._epoch_task: TaskID | None = None
        self._train_task: TaskID | None = None
        self._valid_task: TaskID | None = None
        self._loss_ema: float | None = None
        self._best: tuple[int, float] | None = None

    # context manager: guarantees the live display is torn down on Ctrl-C/errors
    def __enter__(self) -> TrainingProgress:
        """Start the live display (when enabled)."""
        if self.enabled:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                _ThroughputColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[info]}"),
                console=self.console,
                refresh_per_second=8,
            )
            self._progress.start()
            self._epoch_task = self._progress.add_task(
                "epochs", total=self.epochs, info=""
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the live display."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def header(self, settings: dict[str, str]) -> None:
        """Print the run settings as a small panel."""
        body = "\n".join(
            f"[cyan]{key:<10}[/cyan] {value}" for key, value in settings.items()
        )
        self.console.print(Panel(body, title="training run", border_style="blue"))

    def start_epoch(self, epoch: int, train_images: int, valid_images: int) -> None:
        """Add this epoch's train and valid bars."""
        self._loss_ema = None
        if self._progress is None:
            return
        self._train_task = self._progress.add_task(
            f"epoch {epoch}/{self.epochs} train",
            total=train_images,
            info="",
            images=True,
        )
        self._valid_task = self._progress.add_task(
            f"epoch {epoch}/{self.epochs} valid",
            total=valid_images,
            info="",
            images=True,
            visible=False,
        )

    def train_batch(self, images: int, loss: float, lr: float) -> None:
        """Advance the train bar by one batch."""
        self._loss_ema = (
            loss
            if self._loss_ema is None
            else self._LOSS_SMOOTHING * self._loss_ema
            + (1 - self._LOSS_SMOOTHING) * loss
        )
        if self._progress is not None and self._train_task is not None:
            self._progress.update(
                self._train_task,
                advance=images,
                info=f"loss [yellow]{self._loss_ema:.4f}[/yellow]  lr {lr:.2e}",
            )

    def start_valid(self) -> None:
        """Switch the display from the train bar to the valid bar."""
        if self._progress is not None and self._valid_task is not None:
            self._progress.update(self._valid_task, visible=True)

    def valid_batch(self, images: int) -> None:
        """Advance the valid bar by one batch."""
        if self._progress is not None and self._valid_task is not None:
            self._progress.update(self._valid_task, advance=images)

    def end_epoch(self, record: EpochRecord) -> bool:
        """Print the epoch line, update the overall bar; return True if best."""
        is_best = self._best is None or record.monitored > self._best[1]
        if is_best:
            self._best = (record.epoch, record.monitored)
        line = (
            f"epoch {record.epoch:>{len(str(self.epochs))}}/{self.epochs}  "
            f"train {record.train_loss:.4f}  valid {record.val_loss:.4f}  "
            f"acc {record.val_accuracy:.4f}  "
            f"bal-acc {record.val_balanced_accuracy:.4f}  "
            f"macro-F1 {record.val_macro_f1:.4f}  ({record.seconds:.0f}s)"
        )
        if self._progress is not None:
            for task in (self._train_task, self._valid_task):
                if task is not None:
                    self._progress.remove_task(task)
            self._train_task = self._valid_task = None
            assert self._epoch_task is not None  # noqa: S101  # nosec: B101
            best_epoch, best_value = self._best or (record.epoch, record.monitored)
            self._progress.update(
                self._epoch_task,
                advance=1,
                info=f"best {self.monitor} [green]{best_value:.4f}[/green] "
                f"(epoch {best_epoch})",
            )
            style = "green" if is_best else "default"
            marker = "  [bold green]★ best[/bold green]" if is_best else ""
            self._progress.console.print(f"[{style}]{line}[/{style}]{marker}")
        else:
            print(line + ("  * best" if is_best else ""), flush=True)
        return is_best

    def note(self, message: str) -> None:
        """Print a message above the live display."""
        target = self._progress.console if self._progress is not None else self.console
        target.print(message)

    def summary(self, history: list[EpochRecord], best_epoch: int) -> None:
        """Print the per-epoch table, with the best epoch highlighted."""
        table = Table(
            title=f"training summary (best epoch {best_epoch} by {self.monitor})",
            header_style="bold cyan",
        )
        for name in (
            "epoch",
            "train loss",
            "valid loss",
            "acc",
            "bal-acc",
            "macro-F1",
            "lr",
            "time",
        ):
            table.add_column(name, justify="right")
        for record in history:
            table.add_row(
                str(record.epoch),
                f"{record.train_loss:.4f}",
                f"{record.val_loss:.4f}",
                f"{record.val_accuracy:.4f}",
                f"{record.val_balanced_accuracy:.4f}",
                f"{record.val_macro_f1:.4f}",
                f"{record.lr:.2e}",
                f"{record.seconds:.0f}s",
                style="bold green" if record.epoch == best_epoch else None,
            )
        self.console.print(table)
