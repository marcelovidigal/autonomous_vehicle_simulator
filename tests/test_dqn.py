import numpy as np
import pytest

from core import Config, DQNTrainer
from core.config import DQNCfg
from core.dqn import QNet, Replay


def test_qnet_forward_shapes():
    rng = np.random.default_rng(0)
    net = QNet(6, 5, 16, rng)
    q1, h1 = net.forward(np.ones(6, np.float32))
    assert q1.shape == (5,) and h1.shape == (16,)
    q, h = net.forward(np.ones((8, 6), np.float32))
    assert q.shape == (8, 5) and h.shape == (8, 16)


def test_qnet_train_step_reduces_error():
    rng = np.random.default_rng(1)
    net = QNet(4, 3, 24, rng)
    s = rng.normal(0, 1, (128, 4)).astype(np.float32)
    a = rng.integers(0, 3, 128)
    target = rng.normal(0, 1, 128).astype(np.float32)
    first = net.train_step(s, a, target, lr=5e-3)
    for _ in range(300):
        loss = net.train_step(s, a, target, lr=5e-3)
    assert loss < first


def test_replay_push_sample():
    rng = np.random.default_rng(2)
    buf = Replay(100, 3, rng)
    for i in range(250):
        buf.push(np.full(3, i, np.float32), i % 5, float(i), np.zeros(3, np.float32), i % 2)
    assert buf.n == 100
    s, act, r, s2, d = buf.sample(16)
    assert s.shape == (16, 3) and act.shape == (16,) and r.shape == (16,)


def test_dqntrainer_runs_and_records_episodes():
    tr = DQNTrainer(Config(), DQNCfg(seed=0, warmup=100, buffer=2000))
    assert tr.state == "idle"
    tr.step(budget_ms=50)          # noop enquanto idle
    assert tr.episodes == 0
    tr.start()
    for _ in range(60):
        tr.step(budget_ms=40)
    assert tr.episodes >= 1
    assert len(tr.curve) == tr.episodes
    assert np.isfinite(tr.loss)


def test_dqn_save_shapes():
    tr = DQNTrainer(Config(), DQNCfg(seed=0))
    blob = tr.save()
    assert blob["kind"] == "dqn"
    assert len(blob["weights"]) == 4


@pytest.mark.slow
def test_dqn_learns_to_drive_a_lap():
    tr = DQNTrainer(Config(), DQNCfg(seed=0))
    tr.start()
    while tr.episodes < 180:
        tr.step(budget_ms=60)
    assert tr.best_laps >= 1, "o DQN deveria completar ao menos uma volta em ~180 episodios"
    assert tr.best_reward > 20.0
