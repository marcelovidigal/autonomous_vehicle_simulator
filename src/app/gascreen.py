"""Solution 1 screen - Neural Net + Genetic Algorithm."""

from __future__ import annotations

import pygame

from core import Config, Trainer
from core.config import FitnessCfg, GACfg, NetworkCfg, SensorCfg, SimCfg
from core.network import MLP
from core.simulation import n_inputs
from core.track import Track

from . import netview, plot, scene, theme
from .camera import Camera
from .chrome import TOPBAR_H, MenuBar
from .i18n import t
from .widgets import Button, Slider

DEF = Config()


class GAScreen:
    def __init__(self, size):
        self.w, self.h = size
        self.goto: str | None = None
        self.f = pygame.font.Font(None, 19)
        self.big = pygame.font.Font(None, 25)

        self.sim_rect = pygame.Rect(0, TOPBAR_H, 856, self.h - TOPBAR_H)
        self.net_rect = pygame.Rect(872, TOPBAR_H + 12, 416, 330)
        self.plot_rect = pygame.Rect(872, TOPBAR_H + 350, 416, 140)
        self.panel_rect = pygame.Rect(858, TOPBAR_H + 500, 434, self.h - TOPBAR_H - 502)

        self.menu = MenuBar("ga")
        self.turbo = False
        self.track = DEF.sim.track
        self._build_panel()

        self.trainer = Trainer(self._config())
        self.trainer.start()
        self.cam = Camera(self.trainer.track.outer, self.sim_rect)
        self.trace = self.trace_mlp = self.trace_fit = None
        self.tframe = 0

    # ------------------------------------------------------------------ painel
    def _build_panel(self):
        p = self.panel_rect
        x, w, y = p.x + 18, p.w - 36, p.y + 20
        self.sliders: dict[str, Slider] = {}

        def add(key, label, lo, hi, val, integer=False):
            nonlocal y
            self.sliders[key] = Slider(x, y, w, label, lo, hi, val, integer=integer)
            y += 26

        add("hidden", t("ga.sl.hidden"), 0, 16, DEF.network.hidden, integer=True)
        add("sensors", t("ga.sl.sensors"), 3, 9, DEF.sensors.count, integer=True)
        add("population", t("ga.sl.pop"), 10, 80, DEF.ga.population, integer=True)
        add("mrate", t("ga.sl.mrate"), 0.0, 1.0, DEF.ga.mutation_rate)
        add("msigma", t("ga.sl.msigma"), 0.01, 1.0, DEF.ga.mutation_sigma)
        add("elit", t("ga.sl.elit"), 0, 8, DEF.ga.elitism, integer=True)
        add("wspeed", t("ga.sl.wspeed"), 0.0, 1.0, DEF.fitness.w_speed)
        add("crash", t("ga.sl.crash"), 0.0, 5.0, DEF.fitness.crash_penalty)

        bw = (p.w - 36 - 16) // 3
        g = bw + 8
        r1, r2 = y + 6, y + 35
        self.b_play = Button(x, r1, bw, 24, t("btn.play"))
        self.b_gen = Button(x + g, r1, bw, 24, t("btn.nextgen"))
        self.b_turbo = Button(x + 2 * g, r1, bw, 24, t("btn.turbo"))
        self.b_retrain = Button(x, r2, bw, 24, t("btn.retrain"))
        self.b_reset = Button(x + g, r2, bw, 24, t("btn.reset"))
        self.b_track = Button(x + 2 * g, r2, bw, 24, t("btn.track"))
        self.pbuttons = [self.b_play, self.b_gen, self.b_turbo,
                         self.b_retrain, self.b_reset, self.b_track]

    def _config(self) -> Config:
        s = self.sliders
        return Config(
            network=NetworkCfg(hidden=int(s["hidden"].value), activation=DEF.network.activation),
            sensors=SensorCfg(count=int(s["sensors"].value), fov_deg=DEF.sensors.fov_deg,
                              range_u=DEF.sensors.range_u),
            physics=DEF.physics,
            ga=GACfg(population=int(s["population"].value), generations=DEF.ga.generations,
                     mutation_rate=float(s["mrate"].value), mutation_sigma=float(s["msigma"].value),
                     elitism=int(s["elit"].value), tournament_k=DEF.ga.tournament_k,
                     crossover=DEF.ga.crossover, seed=DEF.ga.seed),
            fitness=FitnessCfg(w_progress=DEF.fitness.w_progress, w_speed=float(s["wspeed"].value),
                               w_smooth=DEF.fitness.w_smooth, crash_penalty=float(s["crash"].value),
                               lap_bonus=DEF.fitness.lap_bonus),
            sim=SimCfg(max_steps=DEF.sim.max_steps, track=self.track),
        )

    def _reset_view(self):
        self.cam = Camera(self.trainer.track.outer, self.sim_rect)
        self.trace = self.trace_mlp = self.trace_fit = None

    def _apply(self):
        self.trainer.reset(self._config())
        self.trainer.start()
        self._reset_view()

    # ------------------------------------------------------------------ loop
    def handle(self, ev):
        self.menu.handle(ev)
        for wdg in list(self.sliders.values()) + self.pbuttons:
            wdg.handle(ev)
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                self.trainer.toggle()
            elif ev.key == pygame.K_g:
                self.trainer.advance_generation()
            elif ev.key == pygame.K_t:
                self.turbo = not self.turbo
            elif ev.key == pygame.K_h:
                self.goto = "tutorial_ga"

    def update(self, dt):
        dest = self.menu.poll()
        if dest:
            self.goto = dest
        if self.b_play.poll():
            self.trainer.toggle()
        if self.b_gen.poll():
            self.trainer.advance_generation()
        if self.b_turbo.poll():
            self.turbo = not self.turbo
        if self.b_retrain.poll():
            self._apply()
        if self.b_reset.poll():
            self._build_panel()
            self.turbo = False
            self.track = DEF.sim.track
            self._apply()
        if self.b_track.poll():
            names = Track.list_names()
            self.track = names[(names.index(self.track) + 1) % len(names)]
            self._apply()

        self.trainer.step(budget_ms=28.0 if self.turbo else 7.0)

        if self.trainer.best is not None and self.trainer.best_fitness != self.trace_fit:
            self.trace = self.trainer.best_trace()
            self.trace_mlp = MLP.from_flat(
                self.trainer.best_genome(), n_inputs(self.trainer.cfg), self.trainer.cfg.network
            )
            self.trace_fit = self.trainer.best_fitness
            self.tframe = 0

    def draw(self, surf):
        surf.fill(theme.BG)
        tr = self.trainer
        frame = self.trace[self.tframe % len(self.trace)] if self.trace else None
        pos = frame.pos if frame else None
        head = frame.heading if frame else None
        scene.draw(surf, self.cam, tr.track, pos, head, tr.cfg)

        if frame is not None and self.trace_mlp is not None:
            netview.draw(surf, self.net_rect, frame.act, self.trace_mlp, tr.cfg, self.f)
            self.tframe += 1
        else:
            pygame.draw.rect(surf, theme.PANEL, self.net_rect, border_radius=6)
            surf.blit(self.f.render(t("ga.net.training"), True, theme.TEXT_DIM),
                      (self.net_rect.x + 14, self.net_rect.y + 12))

        plot.draw(surf, self.plot_rect, tr.curve, self.f)
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

    def _draw_hud(self, surf):
        tr = self.trainer
        lines = [
            (self.big, t("ga.hud.gen", g=tr.gen, n=tr.cfg.ga.generations, s=tr.state)),
            (self.f, t("ga.hud.fit", b=tr.best_fitness, m=tr.mean_fitness)),
            (self.f, t("ga.hud.laps", l=tr.best_laps, t=self.track)),
            (self.f, t("ga.hud.keys")),
        ]
        y = TOPBAR_H + 8
        for f, text in lines:
            surf.blit(f.render(text, True, theme.TEXT), (12, y))
            y += f.get_height() + 2
