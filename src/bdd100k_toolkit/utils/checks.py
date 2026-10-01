"""
Loud-failure helpers shared by every adapter.

The official BDD100K toolkit silently produces wrong or empty output in
several places (see ``docs/official-toolkit-review.md``). Adapters here follow
one rule instead: a split that comes out empty is an error, and anything
dropped along the way is counted and reported, never skipped silently.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping


class EmptySplitError(RuntimeError):
    """Raised when a required canonical split ends up with no items."""


def require_nonempty(
    split: str, count: int, detail: str, *, required: bool = True
) -> None:
    """
    Fail (or warn, if not ``required``) when a split has no items.

    Args:
        split: Canonical split name, e.g. ``"train"``.
        count: Number of items actually written to that split.
        detail: What was expected / skipped, appended to the message to
            help diagnose a wrong raw layout.
        required: Raise :class:`EmptySplitError` when True, otherwise only warn
            (used for the small seeded ``valid`` holdout).

    """
    if count > 0:
        return
    message = f"Split '{split}' is empty: {detail}"
    if required:
        raise EmptySplitError(message)
    warnings.warn(message, UserWarning, stacklevel=2)


def format_counts(counts: Mapping[str, int]) -> str:
    """Render ``{"a": 1, "b": 2}`` as ``a=1, b=2`` (empty -> ``none``)."""
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counts.items())
