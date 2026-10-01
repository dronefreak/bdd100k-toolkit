"""Device selection shared by the train/evaluate CLIs."""

from __future__ import annotations

AUTO = "auto"


def resolve_device(device: str | None = AUTO) -> str:
    """
    Resolve ``"auto"`` (or None) to ``"cuda"`` when available, else ``"cpu"``.

    Any explicit value (``"cpu"``, ``"cuda"``, ``"0"``, ``"cuda:1"``, ...) is
    returned unchanged. ``torch`` is imported lazily so importing this module
    stays cheap.
    """
    if device not in (None, AUTO):
        return str(device)
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
