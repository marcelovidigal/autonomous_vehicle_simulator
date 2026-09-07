"""Settings screen: app language and simulation speed (frame-rate cap)."""

from __future__ import annotations

import pygame

from . import i18n, theme
from .chrome import TOPBAR_H, BackBar
from .widgets import Button


class SettingsScreen:
    def __init__(self, size, back: str):
        self.w, self.h = size
        self.goto: str | None = None
        self.back = back
        self.f = pygame.font.Font(None, 21)
        self.f_small = pygame.font.Font(None, 18)
        self.bar = BackBar(i18n.t("set.title"))

        # language row
        self.lang_buttons: list[tuple[str, Button]] = []
        x, y = 80, TOPBAR_H + 90
        for lang in i18n.LANGS:
            self.lang_buttons.append((lang, Button(x, y, 260, 30, i18n.LANG_LABEL[lang])))
            x += 280

        # simulation-speed row
        self.fps_buttons: list[tuple[int, Button]] = []
        x, y = 80, TOPBAR_H + 210
        for fps in i18n.FPS_CHOICES:
            self.fps_buttons.append((fps, Button(x, y, 130, 30, i18n.fps_label(fps))))
            x += 150

    def _all(self):
        return [b for _, b in self.lang_buttons] + [b for _, b in self.fps_buttons]

    def handle(self, ev) -> None:
        self.bar.handle(ev)
        for b in self._all():
            b.handle(ev)
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self.goto = self.back

    def update(self, dt) -> None:
        if self.bar.poll():
            self.goto = self.back
        for lang, b in self.lang_buttons:
            if b.poll() and lang != i18n.get_lang():
                i18n.set_lang(lang)
                self.goto = "settings"  # rebuild with the new language
        for fps, b in self.fps_buttons:
            if b.poll() and fps != i18n.get_fps():
                i18n.set_fps(fps)
                self.goto = "settings"

    def draw(self, surf) -> None:
        surf.fill(theme.BG)

        surf.blit(self.f.render(i18n.t("set.language"), True, theme.TEXT), (80, TOPBAR_H + 52))
        cur_lang = i18n.get_lang()
        for lang, b in self.lang_buttons:
            b.active = lang == cur_lang
            b.draw(surf, self.f_small)
        surf.blit(self.f_small.render(i18n.t("set.language.hint"), True, theme.TEXT_DIM),
                  (80, TOPBAR_H + 128))

        surf.blit(self.f.render(i18n.t("set.speed"), True, theme.TEXT), (80, TOPBAR_H + 172))
        cur_fps = i18n.get_fps()
        for fps, b in self.fps_buttons:
            b.active = fps == cur_fps
            b.draw(surf, self.f_small)
        surf.blit(self.f_small.render(i18n.t("set.speed.hint"), True, theme.TEXT_DIM),
                  (80, TOPBAR_H + 248))

        self.bar.draw(surf, self.w)
