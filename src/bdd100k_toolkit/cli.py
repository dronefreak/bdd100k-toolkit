"""
Unified command-line entry points: ``bdd100k-prepare``, ``-train``, ``-evaluate``.

There is one command per verb, not one per task. The task is inferred from the
dataset key (every registered dataset key belongs to exactly one task), or
given explicitly with ``--task`` (``task=`` for ``bdd100k-train``). The
per-task implementations stay in ``bdd100k_toolkit.scripts`` and are imported
lazily, so ``bdd100k-prepare`` never pays for importing torch.

Examples::

    bdd100k-prepare  --dataset bdd100k-weather --raw-dir RAW --output-dir OUT
    bdd100k-train    dataset=bdd100k-detection dataset.dataset_yaml=.../data.yaml
    bdd100k-evaluate --dataset bdd100k-semantic-seg --checkpoint model.pt ...
"""

from __future__ import annotations

import importlib
import sys
from typing import NoReturn

TASKS = ("classification", "detection", "semantic-seg")
_USAGE_EXIT = 2


def dataset_tasks() -> dict[str, str]:
    """Map every registered dataset key to the task that owns it."""
    from bdd100k_toolkit import classification, detection
    from bdd100k_toolkit.segmentation import semantic

    owners = {
        "classification": classification.list_datasets(),
        "detection": detection.list_datasets(),
        "semantic-seg": semantic.list_datasets(),
    }
    return {key: task for task, keys in owners.items() for key in keys}


def task_for_dataset(key: str) -> str:
    """Return the task owning dataset ``key`` (KeyError listing the valid keys)."""
    tasks = dataset_tasks()
    try:
        return tasks[key]
    except KeyError as err:
        supported = ", ".join(sorted(tasks))
        raise KeyError(f"Unknown dataset '{key}'. Supported: {supported}.") from err


def overview() -> str:
    """Human-readable list of tasks and their datasets."""
    by_task: dict[str, list[str]] = {task: [] for task in TASKS}
    for key, task in sorted(dataset_tasks().items()):
        by_task[task].append(key)
    lines = ["Tasks and datasets:"]
    lines += [f"  {task}: {', '.join(keys)}" for task, keys in by_task.items()]
    return "\n".join(lines)


def pop_option(argv: list[str], name: str) -> tuple[str | None, list[str]]:
    """
    Remove ``name VALUE`` or ``name=VALUE`` from ``argv``.

    Returns the value (None if absent) and the remaining arguments.
    """
    rest: list[str] = []
    value: str | None = None
    items = iter(argv)
    for item in items:
        if item == name:
            value = next(items, None)
            if value is None:
                raise ValueError(f"{name} needs a value")
        elif item.startswith(f"{name}="):
            value = item.split("=", 1)[1]
        else:
            rest.append(item)
    return value, rest


def find_option(argv: list[str], name: str) -> str | None:
    """Return the value of ``name VALUE`` / ``name=VALUE`` without removing it."""
    return pop_option(argv, name)[0]


def _fail(command: str, message: str) -> NoReturn:
    print(f"{command}: error: {message}\n\n{overview()}", file=sys.stderr)
    raise SystemExit(_USAGE_EXIT)


def _resolve_task(
    command: str, explicit: str | None, dataset: str | None, option: str
) -> str:
    """Pick the task from ``--task`` and/or the dataset key, rejecting conflicts."""
    if explicit is not None and explicit not in TASKS:
        _fail(command, f"unknown task '{explicit}' (choose from {', '.join(TASKS)})")
    inferred: str | None = None
    if dataset is not None:
        try:
            inferred = task_for_dataset(dataset)
        except KeyError as err:
            _fail(command, str(err.args[0]))
    if explicit and inferred and explicit != inferred:
        _fail(command, f"dataset '{dataset}' is a {inferred} dataset, not {explicit}")
    task = explicit or inferred
    if task is None:
        _fail(command, f"give the dataset ({option}) or the task")
    return task


def _dispatch(command: str, task: str) -> None:
    module = importlib.import_module(
        f"bdd100k_toolkit.scripts.{command}_{task.replace('-', '_')}"
    )
    module.main()


def _run_argparse_command(command: str) -> None:
    """Dispatch ``prepare`` / ``evaluate``, whose tasks take ``--dataset``."""
    prog = f"bdd100k-{command}"
    argv = sys.argv[1:]
    if not argv or (argv in (["-h"], ["--help"])):
        print(f"usage: {prog} --dataset KEY | --task TASK [options]\n\n{overview()}")
        raise SystemExit(0 if argv else _USAGE_EXIT)
    try:
        explicit, rest = pop_option(argv, "--task")
        dataset = find_option(rest, "--dataset")
    except ValueError as err:
        _fail(prog, str(err))
    task = _resolve_task(prog, explicit, dataset, "--dataset")
    sys.argv = [sys.argv[0], *rest]
    _dispatch(command, task)


def prepare() -> None:
    """Entry point of ``bdd100k-prepare``."""
    _run_argparse_command("prepare")


def evaluate() -> None:
    """Entry point of ``bdd100k-evaluate``."""
    _run_argparse_command("evaluate")


def resolve_train_task(argv: list[str]) -> tuple[str, list[str]]:
    """
    Pick the task for ``bdd100k-train`` from Hydra-style overrides.

    ``task=...`` is consumed here; ``dataset=...`` is left for Hydra but used to
    infer the task. Returns ``(task, remaining_overrides)``.
    """
    explicit, rest = pop_option(argv, "task")
    dataset = find_option(rest, "dataset")
    return _resolve_task("bdd100k-train", explicit, dataset, "dataset="), rest


def train() -> None:
    """Entry point of ``bdd100k-train`` (Hydra overrides, e.g. ``dataset=KEY``)."""
    argv = sys.argv[1:]
    if argv in (["-h"], ["--help"]):
        print(
            "usage: bdd100k-train dataset=KEY | task=TASK [hydra overrides]\n\n"
            + overview()
        )
        raise SystemExit(0)
    task, rest = resolve_train_task(argv)
    sys.argv = [sys.argv[0], *rest]
    _dispatch("train", task)
