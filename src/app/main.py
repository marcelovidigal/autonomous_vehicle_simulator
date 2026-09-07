"""Screen router (desktop and web via pygbag).

Opens on **Solution 1 (Neural Net + GA)** by default. The top bar has the technique tabs
plus About and Settings; the Tutorial button opens the tutorial for the active technique,
and Back returns to it.

Async loop, no threads: `await asyncio.sleep(0)` per frame yields control to the browser.
"""

from __future__ import annotations

import asyncio
import os

import pygame

from . import theme
from .docscreen import DocScreen
from .dqnscreen import DQNScreen
from .gascreen import GAScreen
from .i18n import get_fps, t
from .settingsscreen import SettingsScreen

W, H = 1300, 864


def _make(name: str, size, last_sim: str):
    if name == "ga":
        return GAScreen(size)
    if name == "dqn":
        return DQNScreen(size)
    if name == "about":
        return DocScreen(size, "sobre", t("doc.about"), back=last_sim, with_link=True)
    if name == "settings":
        return SettingsScreen(size, back=last_sim)
    if name == "tutorial_ga":
        return DocScreen(size, "tutorial_ga", t("doc.tutorial_ga"), back="ga", with_pdf=True)
    if name == "tutorial_dqn":
        return DocScreen(size, "tutorial_dqn", t("doc.tutorial_dqn"), back="dqn", with_pdf=True)
    raise ValueError(name)


async def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(t("app.title"))
    clock = pygame.time.Clock()

    # No navegador (pygbag) o canvas so termina de inicializar depois de alguns
    # ciclos do loop async — pintamos algumas vezes cedendo o controle antes de
    # tocar em qualquer trabalho pesado (treino).
    for _ in range(4):
        screen.fill(theme.BG)
        pygame.display.flip()
        await asyncio.sleep(0)

    current = os.environ.get("SMOKE_SCREEN", "ga")
    last_sim = current if current in ("ga", "dqn") else "ga"
    scr = _make(current, (W, H), last_sim)

    smoke = int(os.environ.get("SMOKE_FRAMES", "0"))
    _cyc = os.environ.get("SMOKE_CYCLE", "")
    smoke_cycle = _cyc.split(",") if _cyc else []
    frames = 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            else:
                scr.handle(ev)

        # deixa ~20 frames sem treinar: garante o 1º paint e o sumico da tela de
        # carregamento antes do primeiro lote pesado de simulacao.
        if frames > 20:
            scr.update(1.0 / 60.0)
            if getattr(scr, "goto", None):
                if current in ("ga", "dqn"):
                    last_sim = current
                current = scr.goto
                scr = _make(current, (W, H), last_sim)

        scr.draw(screen)
        pygame.display.flip()
        clock.tick(get_fps())          # Settings > Simulation speed (0 = uncapped)
        await asyncio.sleep(0)

        frames += 1
        if smoke_cycle and frames % 40 == 0:
            current = smoke_cycle.pop(0)
            if current in ("ga", "dqn"):
                last_sim = current
            scr = _make(current, (W, H), last_sim)
        if smoke and frames >= smoke:
            running = False

    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())
