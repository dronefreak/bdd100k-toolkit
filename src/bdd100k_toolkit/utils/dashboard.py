"""
Dashboard picture: the frame, one card per task, and an info strip.

A 1920x1080 canvas (16:9): the frame fitted into the top left, a card per task in
the right column with every class probability, and a strip below the frame with the
models, their latency and the colour legend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from bdd100k_toolkit.utils.draw import (
    AMBER,
    BACKGROUND,
    BAR,
    GREEN,
    MUTED,
    OUTLINE,
    RED,
    SURFACE,
    TEXT,
    TRACK,
    TaskResult,
    accent,
    fit,
    font,
    note,
    percent,
)

if TYPE_CHECKING:
    from bdd100k_toolkit.classification.predict import Prediction

CANVAS, PICTURE_AREA, PANEL_X, PAD = (1920, 1080), (1280, 720), 1280, 16


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    task: str,
    prediction: Prediction,
) -> None:
    """Draw one task: the winner large with a bar, then the other classes as bars."""
    x, y, w, h = box
    color, pad = accent(prediction), 22
    left, right = x + pad + 6, x + w - pad
    draw.rounded_rectangle(
        (x, y, x + w, y + h), 18, fill=SURFACE, outline=OUTLINE, width=2
    )
    draw.rounded_rectangle((x, y + 18, x + 6, y + h - 18), 3, fill=color)

    draw.text((left, y + pad), task.upper(), font=font(15), fill=MUTED)
    if warning := note(prediction):
        warning = fit(warning, 15, right - left - 120)
        draw.text(
            (right - font(15).getlength(warning), y + pad),
            warning,
            font=font(15),
            fill=color,
        )

    label, top = prediction.top1
    shown = percent(top)
    head_y = y + pad + 24
    draw.text(
        (left, head_y), fit(label, 36, right - left - 140), font=font(36), fill=TEXT
    )
    draw.text(
        (right - font(36).getlength(shown), head_y), shown, font=font(36), fill=color
    )
    bar_y = head_y + font(36).getbbox("Ag")[3] + 12
    draw.rounded_rectangle((left, bar_y, right, bar_y + 12), 6, fill=TRACK)
    draw.rounded_rectangle(
        (left, bar_y, left + max(int((right - left) * top), 12), bar_y + 12),
        6,
        fill=color,
    )

    others = prediction.ranked()[1:]
    rows_y = bar_y + 12 + 16
    pitch = min(36, (y + h - pad - rows_y) // max(len(others), 1))
    for i, (name, p) in enumerate(others):
        ry = rows_y + i * pitch
        draw.text(
            (left, ry), fit(name, 16, right - left - 90), font=font(16), fill=TEXT
        )
        draw.text(
            (right - font(16).getlength(percent(p)), ry),
            percent(p),
            font=font(16),
            fill=TEXT,
        )
        by = ry + 20
        draw.rounded_rectangle((left, by, right, by + 6), 3, fill=TRACK)
        if p > 0:
            draw.rounded_rectangle(
                (left, by, left + max(int((right - left) * p), 6), by + 6), 3, fill=BAR
            )


def draw_strip(
    draw: ImageDraw.ImageDraw,
    name: str,
    size: tuple[int, int],
    rows: list[TaskResult],
) -> None:
    """Draw the info strip under the frame: file, model and latency per task, legend."""
    x, y, right = 36, PICTURE_AREA[1], PICTURE_AREA[0] - 36
    draw.text((x, y + 30), "FRAME", font=font(15), fill=MUTED)
    draw.text((x + 90, y + 24), fit(name, 22, 700), font=font(22), fill=TEXT)
    dims = f"{size[0]}x{size[1]}"
    draw.text(
        (right - font(22).getlength(dims), y + 24), dims, font=font(22), fill=MUTED
    )
    draw.line((x, y + 70, right, y + 70), fill=OUTLINE, width=2)

    for text, cx in (("TASK", x + 28), ("MODEL", x + 220)):
        draw.text((cx, y + 84), text, font=font(15), fill=MUTED)
    draw.text(
        (right - font(15).getlength("LATENCY"), y + 84),
        "LATENCY",
        font=font(15),
        fill=MUTED,
    )
    row_h, ry = 38, y + 114
    for task, model, backend, prediction in rows:
        draw.ellipse((x, ry + 5, x + 14, ry + 19), fill=accent(prediction))
        draw.text((x + 28, ry), task, font=font(22), fill=TEXT)
        draw.text(
            (x + 220, ry),
            fit(f"{model}  ({backend})", 22, 700),
            font=font(22),
            fill=TEXT,
        )
        ms = f"{prediction.seconds * 1000:.1f} ms"
        draw.text((right - font(22).getlength(ms), ry), ms, font=font(22), fill=TEXT)
        ry += row_h
    total = sum(r[3].seconds for r in rows)
    draw.line((x + 28, ry - 4, right, ry - 4), fill=TRACK, width=2)
    summary = f"{total * 1000:.1f} ms  ({1 / total:.0f} fps)" if total > 0 else "0 ms"
    draw.text((x + 28, ry + 2), "all models", font=font(22), fill=MUTED)
    draw.text(
        (right - font(22).getlength(summary), ry + 2), summary, font=font(22), fill=TEXT
    )

    ly, cursor = CANVAS[1] - 44, x
    for color, text in (
        (GREEN, "75% or more"),
        (AMBER, "50 to 75%"),
        (RED, "under 50%"),
    ):
        draw.ellipse((cursor, ly + 3, cursor + 14, ly + 17), fill=color)
        draw.text((cursor + 22, ly), text, font=font(17), fill=MUTED)
        cursor += 58 + int(font(17).getlength(text))
    draw.text(
        (cursor, ly), "close call: top two within 15 points", font=font(17), fill=MUTED
    )


def render_dashboard(
    image: Image.Image, name: str, rows: list[TaskResult]
) -> Image.Image:
    """Return the dashboard picture: the frame, one card per task, an info strip."""
    canvas = Image.new("RGB", CANVAS, BACKGROUND)
    area_w, area_h = PICTURE_AREA
    scale = min(area_w / image.width, area_h / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas.paste(
        image.resize(size, Image.Resampling.LANCZOS),
        ((area_w - size[0]) // 2, (area_h - size[1]) // 2),
    )

    draw = ImageDraw.Draw(canvas)
    count = len(rows)
    card_h = min(400, (CANVAS[1] - 2 * PAD - (count - 1) * PAD) // count)
    top = (CANVAS[1] - count * card_h - (count - 1) * PAD) // 2
    for i, (task, _, _, prediction) in enumerate(rows):
        draw_card(
            draw,
            (
                PANEL_X + PAD,
                top + i * (card_h + PAD),
                CANVAS[0] - PANEL_X - 2 * PAD,
                card_h,
            ),
            task,
            prediction,
        )
    draw_strip(draw, name, image.size, rows)
    return canvas
