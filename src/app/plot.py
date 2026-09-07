"""Curve drawn with Pygame primitives (no matplotlib): fitness or reward over time."""

from __future__ import annotations

import pygame

from . import theme
from .i18n import t


def draw(surf, rect, curve, font, caption_key="plot.fitness"):
    pygame.draw.rect(surf, theme.PANEL, rect, border_radius=6)
    pygame.draw.rect(surf, theme.PANEL_LINE, rect, 1, border_radius=6)
    surf.blit(font.render(t(caption_key), True, theme.TEXT_DIM), (rect.x + 10, rect.y + 6))

    if len(curve) < 2:
        return

    pad = 22
    x0, y0 = rect.x + pad, rect.y + pad
    w, h = rect.w - 2 * pad, rect.h - 2 * pad
    lo, hi = min(curve), max(curve)
    span = hi - lo or 1.0

    pygame.draw.line(surf, theme.PANEL_LINE, (x0, y0 + h), (x0 + w, y0 + h), 1)
    pts = [
        (x0 + int(i * w / (len(curve) - 1)), y0 + h - int((v - lo) / span * h))
        for i, v in enumerate(curve)
    ]
    pygame.draw.lines(surf, theme.ACCENT, False, pts, 2)
    surf.blit(font.render(f"{curve[-1]:.2f}", True, theme.TEXT), (x0 + w - 40, y0))
