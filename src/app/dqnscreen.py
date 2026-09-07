"""Solution 2 screen - DQN (Deep Q-Network)."""

from __future__ import annotations

import numpy as np
import pygame

from core import Config, DQNTrainer
from core.config import DQNCfg, SimCfg
from core.track import Track

from . import netview, plot, scene, theme
from .camera import Camera
from .chrome import TOPBAR_H, MenuBar
from .i18n import t
from .widgets import Button, Slider

DDEF = DQNCfg()
_DEF_TRACK = Config().sim.track
_ACT_LABELS = ("<<", "<", "|", ">", ">>")


class _Act:
    __slots__ = ("inputs", "hidden", "outputs")

    def __init__(self, inputs, hidden, outputs):
        self.inputs, self.hidden, self.outputs = inputs, hidden, outputs


class DQNScreen:
    def __init__(self, size):
        self.w, self.h = size
        self.goto: str | None = None
        self.f = pygame.font.Font(None, 19)
        self.big = pygame.font.Font(None, 25)

        self.sim_rect = pygame.Rect(0, TOPBAR_H, 856, self.h - TOPBAR_H)
        self.net_rect = pygame.Rect(872, TOPBAR_H + 12, 416, 330)
        self.plot_rect = pygame.Rect(872, TOPBAR_H + 350, 416, 140)
        self.panel_rect = pygame.Rect(858, TOPBAR_H + 500, 434, self.h - TOPBAR_H - 502)

        self.menu = MenuBar("dqn")
        self.turbo = False
        self.track = _DEF_TRACK
        self._build_panel()

        self.trainer = DQNTrainer(self._cfg(), self._dcfg())
        self.trainer.start()
        self.cam = Camera(self.trainer.track.outer, self.sim_rect)

    def _build_panel(self):
        p = self.panel_rect
        x, w, y = p.x + 18, p.w - 36, p.y + 24
        self.sliders: dict[str, Slider] = {}

        def add(key, label, lo, hi, val, integer=False):
            nonlocal y
            self.sliders[key] = Slider(x, y, w, label, lo, hi, val, integer=integer)
            y += 32

        add("hidden", t("dqn.sl.hidden"), 8, 64, DDEF.hidden, integer=True)
        add("lr", t("dqn.sl.lr"), 0.0002, 0.004, DDEF.lr)
        add("gamma", t("dqn.sl.gamma"), 0.80, 0.999, DDEF.gamma)
        add("edecay", t("dqn.sl.edecay"), 3000, 30000, DDEF.eps_decay_steps, integer=True)

        bw = (p.w - 36 - 16) // 3
        g = bw + 8
        r1, r2 = y + 8, y + 40
        self.b_play = Button(x, r1, bw, 24, t("btn.play"))
        self.b_turbo = Button(x + g, r1, bw, 24, t("btn.turbo"))
        self.b_reset = Button(x + 2 * g, r1, bw, 24, t("btn.reset"))
        self.b_track = Button(x, r2, bw, 24, t("btn.track"))
        self.pbuttons = [self.b_play, self.b_turbo, self.b_reset, self.b_track]

    def _cfg(self) -> Config:
        return Config(sim=SimCfg(max_steps=DDEF.max_steps, track=self.track))

    def _dcfg(self) -> DQNCfg:
        s = self.sliders
        return DQNCfg(
            hidden=int(s["hidden"].value), lr=float(s["lr"].value),
            gamma=float(s["gamma"].value), eps_decay_steps=int(s["edecay"].value),
            max_steps=DDEF.max_steps, seed=DDEF.seed,
        )

    def _apply(self):
        self.trainer.reset(self._cfg(), self._dcfg())
        self.trainer.start()
        self.cam = Camera(self.trainer.track.outer, self.sim_rect)

    # ------------------------------------------------------------------ loop
    def handle(self, ev):
        self.menu.handle(ev)
        for wdg in list(self.sliders.values()) + self.pbuttons:
            wdg.handle(ev)
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                self.trainer.toggle()
            elif ev.key == pygame.K_t:
                self.turbo = not self.turbo
            elif ev.key == pygame.K_h:
                self.goto = "tutorial_dqn"

    def update(self, dt):
        dest = self.menu.poll()
        if dest:
            self.goto = dest
        if self.b_play.poll():
            self.trainer.toggle()
        if self.b_turbo.poll():
            self.turbo = not self.turbo
        if self.b_reset.poll():
            self._build_panel()
            self.turbo = False
            self.track = _DEF_TRACK
            self._apply()
        if self.b_track.poll():
            names = Track.list_names()
            self.track = names[(names.index(self.track) + 1) % len(names)]
            self._apply()

        # treino roda o mais rápido que couber; a visualização anda a ritmo fixo
        self.trainer.step(budget_ms=28.0 if self.turbo else 8.0)
        for _ in range(4 if self.turbo else 1):
            self.trainer.demo_step()

    def draw(self, surf):
        surf.fill(theme.BG)
        tr = self.trainer
        scene.draw(surf, self.cam, tr.track, tr.demo_pos, tr.demo_heading, tr.cfg)

        q = np.asarray(tr.demo_q, float)
        q_norm = q / max(1e-6, np.abs(q).max())
        act = _Act(np.asarray(tr.demo_obs, float), np.asarray(tr.demo_hidden, float), q_norm)
        netview.draw(surf, self.net_rect, act, tr.q, tr.cfg, self.f,
                     out_labels=_ACT_LABELS, caption=t("dqn.net.caption"))

        plot.draw(surf, self.plot_rect, tr.curve, self.f, "plot.reward")
        self._draw_panel(surf)
        self._draw_hud(surf)
        self.menu.draw(surf, self.w)

    def _draw_panel(self, surf):
        pygame.draw.rect(surf, theme.PANEL, self.panel_rect, border_radius=6)
        pygame.draw.rect(surf, theme.PANEL_LINE, self.panel_rect, 1, border_radius=6)
        for s in self.sliders.values():
            s.draw(surf, self.f)
        self.b_play.active = self.trainer.state == "running"
        self.b_turbo.active = self.turbo
        for b in self.pbuttons:
            b.draw(surf, self.f)
        surf.blit(self.f.render(t("dqn.hint.apply"), True, theme.TEXT_DIM),
                  (self.panel_rect.x + 18, self.panel_rect.bottom - 22))

    def _draw_hud(self, surf):
        tr = self.trainer
        greedy = _ACT_LABELS[int(np.argmax(tr.demo_q))]
        lines = [
            (self.big, t("dqn.hud.ep", e=tr.episodes, s=tr.state)),
            (self.f, t("dqn.hud.best", r=tr.best_reward, l=tr.best_laps)),
            (self.f, t("dqn.hud.stats", e=tr.eps, ls=tr.loss, b=tr.buf.n)),
            (self.f, t("dqn.hud.greedy", a=greedy, t=self.track)),
            (self.f, t("dqn.hud.keys")),
        ]
        y = TOPBAR_H + 8
        for f, text in lines:
            surf.blit(f.render(text, True, theme.TEXT), (12, y))
            y += f.get_height() + 2
