"""
Overlay picture: predictions drawn on the image in a small frosted-glass panel.

The picture behind the panel is blurred and darkened so it reads on bright and dark
frames alike. Everything scales with the image width.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter

from bdd100k_toolkit.utils.draw import (
    TaskResult,
    accent,
    fit,
    font,
    percent,
    short_note,
)

MIN_SIZE = 0.5  # overlay size at size=0, relative to the original (first) design
FAKE_BOLD_FROM = 20  # font size (px) from which names are drawn slightly bolder


def render_overlay(
    image: Image.Image, rows: list[TaskResult], size: float = 0.0
) -> Image.Image:
    """
    Return ``image`` with each task's prediction drawn on it in a frosted-glass panel.

    The panel sits top left (the picture behind it is blurred and darkened, so it
    reads on bright and dark frames alike); a small pill bottom right shows the
    total latency. Sizes scale with the image width, so any resolution looks alike,
    and ``size`` (0 to 1) sets how much of the picture the overlay takes: 0 is the
    default compact overlay, 1 is twice that (the original design).
    """
    base = max(image.width / 1280, 0.4)  # 1.0 for a 1280 px wide frame
    scale = base * (MIN_SIZE + (1 - MIN_SIZE) * size)

    def px(value: float) -> int:
        return max(1, round(value * scale))

    def text_px(value: float, floor: int) -> int:
        """Return a font size that scales with the overlay but stays readable."""
        return max(round(floor * base), px(value))

    label_px, name_px, note_px = text_px(13, 10), text_px(28, 15), text_px(12, 9)
    bar_h, gap, pad, margin = px(5), px(14), px(22), px(24)
    name_y, bar_y = label_px + px(4), label_px + px(4) + name_px + px(8)
    row_h = bar_y + bar_h + gap
    width = min(px(440), int(image.width * 0.92))
    height = 2 * pad + row_h * len(rows) - gap
    x0 = y0 = margin
    canvas = image.convert("RGBA")

    # frosted glass: blur what is behind the panel, then tint it
    box = (x0, y0, x0 + width, y0 + height)
    glass = Image.alpha_composite(
        canvas.crop(box).filter(ImageFilter.GaussianBlur(px(16))),
        Image.new("RGBA", (width, height), (8, 12, 18, 168)),
    )
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width, height), px(20), fill=255)
    canvas.paste(glass, (x0, y0), mask)

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(box, px(20), outline=(255, 255, 255, 48), width=1)
    left, right = x0 + pad, x0 + width - pad
    for index, (task, _, _, prediction) in enumerate(rows):
        ry = y0 + pad + index * row_h
        if index:  # hairline between rows
            draw.line(
                (left, ry - gap // 2 - 1, right, ry - gap // 2 - 1),
                fill=(255, 255, 255, 28),
                width=1,
            )
        color = accent(prediction)
        label, top = prediction.top1

        dot = max(label_px - 3, 4)
        draw.ellipse(
            (
                left,
                ry + (label_px - dot) // 2 + 1,
                left + dot,
                ry + (label_px - dot) // 2 + 1 + dot,
            ),
            fill=color,
        )
        draw.text(
            (left + dot + px(10), ry),
            task.upper(),
            font=font(label_px),
            fill=(190, 198, 208),
        )
        if warning := short_note(prediction):
            warning = fit(
                warning,
                note_px,
                right - left - font(label_px).getlength(task.upper()) - px(40),
            )
            draw.text(
                (right - font(note_px).getlength(warning), ry + 1),
                warning,
                font=font(note_px),
                fill=color,
            )

        shown = percent(top)
        draw.text(
            (right - font(name_px).getlength(shown), ry + name_y),
            shown,
            font=font(name_px),
            fill=color,
            stroke_width=int(name_px >= FAKE_BOLD_FROM),
            stroke_fill=color,
        )
        name = fit(
            label, name_px, right - left - font(name_px).getlength(shown) - px(16)
        )
        draw.text(
            (left, ry + name_y),
            name,
            font=font(name_px),
            fill=(244, 247, 250),
            stroke_width=int(name_px >= FAKE_BOLD_FROM),
            stroke_fill=(244, 247, 250),
        )

        by = ry + bar_y
        draw.rounded_rectangle(
            (left, by, right, by + bar_h), bar_h // 2, fill=(255, 255, 255, 40)
        )
        draw.rounded_rectangle(
            (left, by, left + max(int((right - left) * top), bar_h), by + bar_h),
            bar_h // 2,
            fill=color,
        )

    # latency pill, bottom right
    total = sum(r[3].seconds for r in rows)
    text = f"{total * 1000:.0f} ms" + (f"  ·  {1 / total:.0f} fps" if total > 0 else "")
    pill_font = font(text_px(15, 11))
    pill_pad = px(14)
    pill_w, pill_h = (
        int(pill_font.getlength(text)) + 2 * pill_pad,
        text_px(15, 11) + px(16),
    )
    px0, py0 = image.width - margin - pill_w, image.height - margin - pill_h
    draw.rounded_rectangle(
        (px0, py0, px0 + pill_w, py0 + pill_h),
        pill_h // 2,
        fill=(8, 12, 18, 170),
        outline=(255, 255, 255, 40),
    )
    draw.text(
        (px0 + pill_pad, py0 + (pill_h - text_px(15, 11)) // 2 - 1),
        text,
        font=pill_font,
        fill=(214, 220, 228),
    )
    return Image.alpha_composite(canvas, layer).convert("RGB")
