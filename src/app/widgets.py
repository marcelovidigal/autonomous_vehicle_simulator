"""Widgets mínimos desenhados no canvas (leves e compatíveis com pygbag)."""

from __future__ import annotations

import pygame

from . import theme


class Slider:
    def __init__(self, x, y, w, label, lo, hi, value, integer=False):
        self.rect = pygame.Rect(x, y, w, 14)
        self.label = label
        self.lo, self.hi = float(lo), float(hi)
        self.integer = integer
        self.value = int(round(value)) if integer else float(value)
        self._drag = False

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and self._hit(ev.pos):
            self._drag = True
            self._set_from_x(ev.pos[0])
        elif ev.type == pygame.MOUSEBUTTONUP:
            self._drag = False
        elif ev.type == pygame.MOUSEMOTION and self._drag:
            self._set_from_x(ev.pos[0])

    def _hit(self, pos):
        return self.rect.inflate(12, 16).collidepoint(pos)

    def _set_from_x(self, mx):
        t = min(1.0, max(0.0, (mx - self.rect.x) / self.rect.w))
        v = self.lo + t * (self.hi - self.lo)
        self.value = int(round(v)) if self.integer else round(v, 3)

    def draw(self, surf, font):
        cy = self.rect.centery
        pygame.draw.line(surf, theme.PANEL_LINE, (self.rect.x, cy), (self.rect.right, cy), 3)
        t = (self.value - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0.0
        hx = int(self.rect.x + t * self.rect.w)
        pygame.draw.circle(surf, theme.ACCENT, (hx, cy), 6)
        txt = f"{self.label}: {self.value:g}"
        surf.blit(font.render(txt, True, theme.TEXT), (self.rect.x, self.rect.y - 15))


class Button:
    def __init__(self, x, y, w, h, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self._clicked = False
        self.active = False

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            self._clicked = True

    def poll(self) -> bool:
        c = self._clicked
        self._clicked = False
        return c

    def draw(self, surf, font):
        bg = theme.ACCENT if self.active else theme.PANEL_LINE
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(surf, theme.PANEL_LINE, self.rect, 1, border_radius=4)
        col = theme.BG if self.active else theme.TEXT
        t = font.render(self.label, True, col)
        surf.blit(t, t.get_rect(center=self.rect.center))
