"""
Shared look and small drawing helpers for the demo pictures.

Colours, confidence thresholds, a cached font, text fitting, and the wording used
for close calls. Every demo renderer (overlay, dashboard, and later detection or
segmentation views) draws with these, so they all agree on how a result looks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

from PIL import ImageFont

if TYPE_CHECKING:
    from bdd100k_toolkit.classification.predict import Prediction

BACKGROUND, SURFACE, OUTLINE = (18, 22, 28), (25, 31, 39), (42, 51, 62)
TEXT, MUTED, TRACK, BAR = (232, 236, 241), (139, 149, 161), (35, 42, 51), (91, 107, 124)
GREEN, AMBER, RED = (61, 220, 132), (255, 176, 32), (255, 93, 93)
CONFIDENT, UNSURE, CLOSE = 0.75, 0.50, 0.15  # accent thresholds; "close call" margin


class TaskResult(NamedTuple):
    """One task's output for a frame, with where it came from."""

    task: str
    model: str
    backend: str
    prediction: Prediction


@lru_cache(maxsize=32)
def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the scalable default font at ``size`` px."""
    return ImageFont.load_default(size=size)


def fit(text: str, size: int, width: float) -> str:
    """Shorten ``text`` with an ellipsis until it fits in ``width`` pixels."""
    if font(size).getlength(text) <= width:
        return text
    while text and font(size).getlength(text + "...") > width:
        text = text[:-1]
    return text + "..."


def percent(p: float) -> str:
    """Format a probability, e.g. ``93.2%`` (``<0.1%`` for tiny ones)."""
    return "<0.1%" if 0 < p < 0.001 else f"{p * 100:.1f}%"  # noqa: PLR2004


def accent(prediction: Prediction) -> tuple[int, int, int]:
    """Green, amber or red by the confidence of the top class."""
    top = prediction.top1[1]
    return GREEN if top >= CONFIDENT else AMBER if top >= UNSURE else RED


def note(prediction: Prediction) -> str | None:
    """Warn when the winner is not clearly ahead (names the runner-up)."""
    ranked = prediction.ranked()
    if len(ranked) > 1 and prediction.margin < CLOSE:
        # subtract the rounded values so the note agrees with the percentages shown
        gap = round(ranked[0][1] * 100, 1) - round(ranked[1][1] * 100, 1)
        return f"close call: {ranked[1][0]} is {gap:.1f} pts behind"
    return f"low confidence ({ranked[0][1]:.0%})" if ranked[0][1] < UNSURE else None


def short_note(prediction: Prediction) -> str | None:
    """Return a compact warning for the overlay: the runner-up of a close call."""
    ranked = prediction.ranked()
    if len(ranked) > 1 and prediction.margin < CLOSE:
        return f"vs {ranked[1][0]}"
    return "low confidence" if ranked[0][1] < UNSURE else None
