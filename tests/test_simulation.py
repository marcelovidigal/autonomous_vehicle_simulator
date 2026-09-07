import numpy as np
import pytest

from core import Config, Track
from core.config import GACfg
from core.network import MLP
from core.simulation import (
    n_inputs,
    simulate,
    simulate_population,
)


def _rand_pop(cfg, p, seed=0):
    k = MLP.n_params(n_inputs(cfg), cfg.network)
    return np.random.default_rng(seed).normal(0, 0.5, (p, k)).astype(np.float32)


def test_simulate_returns_finite_and_trace():
    cfg = Config()
    r = simulate(_rand_pop(cfg, 1)[0], Track.load("circuito_1"), cfg, record=True)
    assert np.isfinite(r.fitness)
    assert r.trace and r.trace[0].act.outputs.shape == (2,)
    assert r.state.steps >= 1


def test_stationary_genome_is_stalled_not_infinite():
    cfg = Config()
    k = MLP.n_params(n_inputs(cfg), cfg.network)
    r = simulate(np.zeros(k, np.float32), Track.load("circuito_1"), cfg)
    assert r.state.steps < cfg.sim.max_steps  # encerrado por stall


def test_vector_matches_scalar():
    cfg = Config(ga=GACfg(population=12))
    track = Track.load("circuito_1")
    pop = _rand_pop(cfg, 12, seed=3)
    vec, _ = simulate_population(pop, track, cfg)
    scal = np.array([simulate(g, track, cfg).fitness for g in pop])
    assert np.max(np.abs(vec - scal)) < 0.05


@pytest.mark.parametrize("name", Track.list_names())
def test_population_runs_on_every_track(name):
    cfg = Config()
    pop = _rand_pop(cfg, 16, seed=1)
    scores, laps = simulate_population(pop, Track.load(name), cfg)
    assert scores.shape == (16,) and laps.shape == (16,)
    assert np.all(np.isfinite(scores))
