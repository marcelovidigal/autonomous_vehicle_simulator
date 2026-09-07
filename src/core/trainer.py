"""Orquestra o treino de forma incremental (sem threads), pronto para o loop async."""

from __future__ import annotations

import time

import numpy as np

from .config import Config
from .evolution import init_population, next_population
from .network import MLP
from .simulation import n_inputs, simulate_population
from .track import Track


class Trainer:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.reset()

    # ------------------------------------------------------------------ estado
    def reset(self, cfg: Config | None = None) -> None:
        if cfg is not None:
            self.cfg = cfg
        self.rng = np.random.default_rng(self.cfg.ga.seed)
        self.track = Track.load(self.cfg.sim.track)
        self.n_params = MLP.n_params(n_inputs(self.cfg), self.cfg.network)
        self.pop = init_population(self.cfg.ga.population, self.n_params, self.rng)
        self.gen = 0
        self.state = "idle"                 # idle | running | done
        self.best: np.ndarray | None = None
        self.best_fitness = -1e9
        self.best_laps = 0
        self.mean_fitness = 0.0
        self.curve: list[float] = []
        self._fit = np.full(self.cfg.ga.population, np.nan)
        self._laps = np.zeros(self.cfg.ga.population, dtype=int)
        self._cursor = 0

    # ------------------------------------------------------------------ controle
    def start(self) -> None:
        if self.state != "done":
            self.state = "running"

    def pause(self) -> None:
        if self.state == "running":
            self.state = "idle"

    def toggle(self) -> None:
        self.pause() if self.state == "running" else self.start()

    # ------------------------------------------------------------------ avanço
    def step(self, budget_ms: float = 8.0, batch: int = 8) -> None:
        """Avança o treino pelo tempo dado, avaliando a população em sub-lotes."""
        if self.state != "running":
            return
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) * 1000.0 < budget_ms:
            self._eval_batch(batch)
            if self._cursor >= self.cfg.ga.population:
                self._finish_generation()
                if self.state == "done":
                    return

    def advance_generation(self) -> None:
        """Roda a geração inteira de uma vez (botão '+1 Geração' / modo turbo)."""
        while self._cursor < self.cfg.ga.population:
            self._eval_batch(self.cfg.ga.population)
        self._finish_generation()

    def _eval_batch(self, batch: int) -> None:
        end = min(self._cursor + batch, self.cfg.ga.population)
        scores, laps = simulate_population(self.pop[self._cursor:end], self.track, self.cfg)
        self._fit[self._cursor:end] = scores
        self._laps[self._cursor:end] = laps
        self._cursor = end

    def _finish_generation(self) -> None:
        fit = self._fit
        b = int(np.argmax(fit))
        if fit[b] > self.best_fitness:
            self.best_fitness = float(fit[b])
            self.best = self.pop[b].copy()
            self.best_laps = int(self._laps[b])
        self.mean_fitness = float(np.mean(fit))
        self.curve.append(float(fit[b]))
        self.gen += 1
        if self.gen >= self.cfg.ga.generations:
            self.state = "done"
        else:
            self.pop = next_population(self.pop, fit, self.cfg.ga, self.rng)
        self._fit = np.full(self.cfg.ga.population, np.nan)
        self._laps = np.zeros(self.cfg.ga.population, dtype=int)
        self._cursor = 0

    # ------------------------------------------------------------------ auxiliares
    def best_genome(self) -> np.ndarray | None:
        if self.best is not None:
            return self.best
        return self.pop[0] if len(self.pop) else None

    def best_trace(self):
        from .simulation import simulate

        g = self.best_genome()
        return simulate(g, self.track, self.cfg, record=True).trace if g is not None else None

    def save(self) -> dict:
        return {
            "config": self.cfg.to_dict(),
            "genome": np.asarray(self.best_genome()).tolist(),
            "generation": self.gen,
            "best_fitness": self.best_fitness,
        }

    def load(self, blob: dict) -> None:
        self.reset(Config.from_dict(blob["config"]))
        g = np.asarray(blob["genome"], np.float32)
        if g.size == self.n_params:
            self.best = g
            self.pop[0] = g
