"""Top menu bar (technique tabs + Settings + Tutorial) and a simple Back bar."""

from __future__ import annotations

import pygame

from . import theme
from .i18n import t
from .widgets import Button

TOPBAR_H = 40
_TAB_KEYS = ("ga", "dqn", "about", "settings")
_TAB_STR = {"ga": "menu.ga", "dqn": "menu.dqn", "about": "menu.about", "settings": "menu.settings"}


class MenuBar:
    """Tabs GA / DQN / About / Settings + a Tutorial button for the active technique."""

    def __init__(self, active: str):
        self.active = active
        self.f = pygame.font.Font(None, 20)
        self._tabs: list[tuple[str, Button]] = []
        x = 12
        for key in _TAB_KEYS:
            label = t(_TAB_STR[key])
            w = self.f.size(label)[0] + 22
            self._tabs.append((key, Button(x, 7, w, 26, label)))
            x += w + 5
        self.b_tut = Button(0, 7, 116, 26, t("menu.tutorial"))
        self.show_tut = active in ("ga", "dqn")

    def handle(self, ev) -> None:
        for _k, b in self._tabs:
            b.handle(ev)
        if self.show_tut:
            self.b_tut.handle(ev)

    def poll(self) -> str | None:
        for key, b in self._tabs:
            if b.poll() and key != self.active:
                return key
        if self.show_tut and self.b_tut.poll():
            return f"tutorial_{self.active}"
        return None

    def draw(self, surf, width: int) -> None:
        pygame.draw.rect(surf, theme.PANEL, (0, 0, width, TOPBAR_H))
        pygame.draw.line(surf, theme.PANEL_LINE, (0, TOPBAR_H), (width, TOPBAR_H), 1)
        for key, b in self._tabs:
            b.active = key == self.active
            b.draw(surf, self.f)
        if self.show_tut:
            self.b_tut.rect.topleft = (width - self.b_tut.rect.w - 12, 7)
            self.b_tut.draw(surf, self.f)


class BackBar:
    """Simple bar with a Back button + title (Tutorial / About / Settings screens)."""

    def __init__(self, title: str):
        self.title = title
        self.f = pygame.font.Font(None, 20)
        self.b_back = Button(12, 7, 116, 26, t("nav.back"))

    def handle(self, ev) -> None:
        self.b_back.handle(ev)

    def poll(self) -> bool:
        return self.b_back.poll()

    def draw(self, surf, width: int) -> None:
        pygame.draw.rect(surf, theme.PANEL, (0, 0, width, TOPBAR_H))
        pygame.draw.line(surf, theme.PANEL_LINE, (0, TOPBAR_H), (width, TOPBAR_H), 1)
        self.b_back.draw(surf, self.f)
        tx = self.f.render(self.title, True, theme.TEXT)
        surf.blit(tx, (width // 2 - tx.get_width() // 2, 11))
