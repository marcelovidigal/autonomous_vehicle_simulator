"""Algoritmo genético sobre o vetor plano de pesos: elitismo, torneio, crossover, mutação."""

from __future__ import annotations

import numpy as np

from .config import GACfg


def init_population(pop: int, n_params: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 0.5, (pop, n_params)).astype(np.float32)


def _tournament(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    picks = rng.integers(0, len(fitness), size=max(1, k))
    return int(picks[np.argmax(fitness[picks])])


def next_population(genomes: np.ndarray, fitness: np.ndarray, cfg: GACfg,
                    rng: np.random.Generator) -> np.ndarray:
    pop, n = genomes.shape
    order = np.argsort(fitness)[::-1]
    nxt = np.empty_like(genomes)

    elite = min(cfg.elitism, pop)
    nxt[:elite] = genomes[order[:elite]]

    for i in range(elite, pop):
        p1 = genomes[_tournament(fitness, cfg.tournament_k, rng)]
        if cfg.crossover:
            p2 = genomes[_tournament(fitness, cfg.tournament_k, rng)]
            take_p1 = rng.random(n) < 0.5
            child = np.where(take_p1, p1, p2).astype(np.float32)
        else:
            child = p1.copy()
        mutate = rng.random(n) < cfg.mutation_rate
        child[mutate] += rng.normal(0.0, cfg.mutation_sigma, int(mutate.sum())).astype(np.float32)
        nxt[i] = child
    return nxt
