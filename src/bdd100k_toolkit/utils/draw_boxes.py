"""
Detection overlay for the demos: boxes with labels, a class-count panel, a latency pill.

Same look as the classification overlay (translucent dark panels, rounded pill), and
everything scales with the image width.
"""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from bdd100k_toolkit.utils.draw import font

if TYPE_CHECKING:
    from bdd100k_toolkit.detection.predict import Detections

PANEL, PANEL_OUTLINE, TEXT = (8, 12, 18, 170), (255, 255, 255, 40), (232, 236, 241)


def class_color(index: int, count: int) -> tuple[int, int, int]:
    """Return a distinct, bright colour for class ``index`` of ``count`` classes."""
    red, green, blue = colorsys.hsv_to_rgb(index / max(count, 1), 0.65, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def render_boxes(
    image: Image.Image, detections: Detections, *, beside_overlay: bool = False
) -> Image.Image:
    """
    Return ``image`` with every detection drawn on it.

    With ``beside_overlay`` the class counts move to the top right and the latency
    pill to the bottom left, leaving the corners the classification overlay uses.
    """
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = max(image.width / 1280, 0.4)  # 1.0 for a 1280 px wide frame
    names, total = detections.class_names, len(detections.class_names)
    line, label_px = max(1, round(2 * scale)), max(10, round(13 * scale))

    # small boxes first, so large ones never hide them
    order = sorted(
        range(len(detections.boxes)),
        key=lambda i: (
            -(detections.boxes[i][2] - detections.boxes[i][0])
            * (detections.boxes[i][3] - detections.boxes[i][1])
        ),
    )
    for i in order:
        x1, y1, x2, y2 = detections.boxes[i]
        color = class_color(detections.labels[i], total)
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=line)
        if x2 - x1 < 40 * scale:  # too small to carry a readable label
            continue
        text = f"{names[detections.labels[i]]} {detections.scores[i]:.0%}"
        chip_w = int(font(label_px).getlength(text)) + round(8 * scale)
        chip_h = label_px + round(6 * scale)
        top = y1 - chip_h if y1 - chip_h >= 0 else y1
        draw.rectangle((x1, top, x1 + chip_w, top + chip_h), fill=(*color, 215))
        draw.text(
            (x1 + round(4 * scale), top + round(2 * scale)),
            text,
            font=font(label_px),
            fill=(10, 14, 20),
        )

    _draw_counts(draw, detections, scale, image.width if beside_overlay else None)
    _draw_latency(draw, image.size, detections.seconds, scale, beside_overlay)
    return Image.alpha_composite(canvas, layer).convert("RGB")


def _draw_counts(
    draw: ImageDraw.ImageDraw,
    detections: Detections,
    scale: float,
    right_edge: int | None = None,
) -> None:
    """Top panel (left, or right if ``right_edge`` is given): one row per class."""
    counts = detections.counts().most_common()
    if not counts:
        return
    row_px, pad = max(11, round(15 * scale)), round(12 * scale)
    row_h, margin = row_px + round(8 * scale), round(16 * scale)
    width = (
        max(int(font(row_px).getlength(f"{name}  {n}")) for name, n in counts)
        + 2 * pad
        + row_px
    )
    height = len(counts) * row_h + 2 * pad - round(4 * scale)
    left = margin if right_edge is None else right_edge - margin - width
    draw.rounded_rectangle(
        (left, margin, left + width, margin + height),
        round(10 * scale),
        fill=PANEL,
        outline=PANEL_OUTLINE,
    )
    names = detections.class_names
    for k, (name, n) in enumerate(counts):
        y = margin + pad + k * row_h
        color = class_color(names.index(name), len(names))
        dot = row_px // 2
        draw.ellipse((left + pad, y + 2, left + pad + dot, y + 2 + dot), fill=color)
        draw.text(
            (left + pad + dot + round(8 * scale), y),
            f"{name}  {n}",
            font=font(row_px),
            fill=TEXT,
        )


def _draw_latency(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    seconds: float,
    scale: float,
    left_side: bool = False,
) -> None:
    """Bottom pill (right, or left if ``left_side``) with the time and frame rate."""
    text = f"{seconds * 1000:.0f} ms" + (
        f"  ·  {1 / seconds:.0f} fps" if seconds > 0 else ""
    )
    px = max(11, round(15 * scale))
    pad, margin = round(14 * scale), round(16 * scale)
    pill_w, pill_h = int(font(px).getlength(text)) + 2 * pad, px + round(16 * scale)
    x0 = margin if left_side else size[0] - margin - pill_w
    y0 = size[1] - margin - pill_h
    draw.rounded_rectangle(
        (x0, y0, x0 + pill_w, y0 + pill_h),
        pill_h // 2,
        fill=PANEL,
        outline=PANEL_OUTLINE,
    )
    draw.text((x0 + pad, y0 + (pill_h - px) // 2 - 1), text, font=font(px), fill=TEXT)
