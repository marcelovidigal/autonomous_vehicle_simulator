"""Transformação mundo -> tela, ajustando a pista dentro de um retângulo."""

from __future__ import annotations

import numpy as np


class Camera:
    def __init__(self, world_pts, screen_rect, margin: int = 36):
        p = np.asarray(world_pts, float)
        self.min = p.min(axis=0)
        span = p.max(axis=0) - self.min
        span[span == 0] = 1.0
        sr = screen_rect
        self.scale = min((sr.width - 2 * margin) / span[0], (sr.height - 2 * margin) / span[1])
        self.ox = sr.x + margin + (sr.width - 2 * margin - span[0] * self.scale) / 2
        self.oy = sr.y + margin + (sr.height - 2 * margin - span[1] * self.scale) / 2

    def to_screen(self, pt) -> tuple[int, int]:
        return (
            int(self.ox + (pt[0] - self.min[0]) * self.scale),
            int(self.oy + (pt[1] - self.min[1]) * self.scale),
        )

    def poly(self, pts) -> list[tuple[int, int]]:
        return [self.to_screen(p) for p in pts]
