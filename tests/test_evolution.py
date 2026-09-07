import numpy as np

from core.config import GACfg
from core.evolution import init_population, next_population


def _fitness_toy(genomes):
    # fitness = -soma dos quadrados (ótimo em zero)
    return -np.sum(genomes**2, axis=1)


def test_init_shape_and_dtype():
    pop = init_population(20, 12, np.random.default_rng(0))
    assert pop.shape == (20, 12)
    assert pop.dtype == np.float32


def test_next_population_shape_preserved():
    rng = np.random.default_rng(0)
    pop = init_population(20, 12, rng)
    nxt = next_population(pop, _fitness_toy(pop), GACfg(population=20), rng)
    assert nxt.shape == pop.shape


def test_elitism_preserves_best():
    rng = np.random.default_rng(1)
    pop = init_population(30, 10, rng)
    fit = _fitness_toy(pop)
    best = pop[int(np.argmax(fit))]
    nxt = next_population(pop, fit, GACfg(population=30, elitism=2), rng)
    assert np.array_equal(nxt[0], best)


def test_deterministic_given_seed():
    a_rng, b_rng = np.random.default_rng(7), np.random.default_rng(7)
    pa = init_population(16, 8, a_rng)
    pb = init_population(16, 8, b_rng)
    na = next_population(pa, _fitness_toy(pa), GACfg(population=16), a_rng)
    nb = next_population(pb, _fitness_toy(pb), GACfg(population=16), b_rng)
    assert np.array_equal(na, nb)


def test_ga_improves_toy_objective():
    rng = np.random.default_rng(0)
    cfg = GACfg(population=40, mutation_rate=0.2, mutation_sigma=0.15, elitism=2)
    pop = init_population(40, 10, rng)
    first = _fitness_toy(pop).max()
    for _ in range(30):
        pop = next_population(pop, _fitness_toy(pop), cfg, rng)
    assert _fitness_toy(pop).max() > first
