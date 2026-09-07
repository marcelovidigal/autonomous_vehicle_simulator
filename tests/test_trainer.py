import numpy as np

from core import Config, Trainer
from core.config import GACfg


def _small():
    return Config(ga=GACfg(population=12, generations=6, seed=2))


def test_idle_step_is_noop():
    tr = Trainer(_small())
    assert tr.state == "idle"
    tr.step(budget_ms=50)
    assert tr.gen == 0


def test_advance_generation_progresses():
    tr = Trainer(_small())
    tr.advance_generation()
    tr.advance_generation()
    assert tr.gen == 2
    assert len(tr.curve) == 2
    assert tr.best is not None


def test_reaches_done_state():
    tr = Trainer(_small())
    tr.start()
    for _ in range(200):
        tr.step(budget_ms=50)
        if tr.state == "done":
            break
    assert tr.state == "done"
    assert tr.gen == 6


def test_step_and_advance_give_same_curve():
    a = Trainer(_small())
    a.start()
    while a.state != "done":
        a.step(budget_ms=1, batch=3)
    b = Trainer(_small())
    while b.state != "done":
        b.advance_generation()
    assert np.allclose(a.curve, b.curve)


def test_save_load_roundtrip():
    tr = Trainer(_small())
    tr.advance_generation()
    blob = tr.save()
    other = Trainer()
    other.load(blob)
    assert other.cfg == tr.cfg
    assert np.allclose(np.asarray(other.best), np.asarray(tr.best_genome()))


def test_reset_restores_initial_state():
    tr = Trainer(_small())
    tr.advance_generation()
    tr.reset()
    assert tr.gen == 0 and tr.best is None and tr.curve == []
