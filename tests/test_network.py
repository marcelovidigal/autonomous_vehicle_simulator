import numpy as np
import pytest

from core import MLP
from core.config import NetworkCfg


@pytest.mark.parametrize("hidden", [0, 1, 6, 16])
def test_n_params_matches_flat_size(hidden):
    cfg = NetworkCfg(hidden=hidden)
    m = MLP(6, cfg)
    assert m.to_flat().size == MLP.n_params(6, cfg)


@pytest.mark.parametrize("hidden", [0, 6])
def test_forward_shape_and_bounds(hidden):
    m = MLP(6, NetworkCfg(hidden=hidden))
    out = m.forward(np.ones(6, np.float32))
    assert out.shape == (2,)
    assert np.all(np.abs(out) <= 1.0)
    assert m.last is not None and m.last[0].shape == (6,)


def test_flat_roundtrip():
    cfg = NetworkCfg(hidden=8)
    m = MLP(5, cfg)
    vec = np.arange(MLP.n_params(5, cfg), dtype=np.float32)
    m.load_flat(vec)
    assert np.array_equal(m.to_flat(), vec)


def test_deterministic_given_seed():
    cfg = NetworkCfg(hidden=6)
    a = MLP(6, cfg, np.random.default_rng(3)).to_flat()
    b = MLP(6, cfg, np.random.default_rng(3)).to_flat()
    assert np.array_equal(a, b)


def test_relu_activation_changes_output():
    x = np.linspace(-1, 1, 6, dtype=np.float32)
    n = MLP.n_params(6, NetworkCfg(hidden=6))
    vec = np.random.default_rng(0).normal(0, 1, n).astype(np.float32)
    t = MLP.from_flat(vec, 6, NetworkCfg(hidden=6, activation="tanh")).forward(x)
    r = MLP.from_flat(vec, 6, NetworkCfg(hidden=6, activation="relu")).forward(x)
    assert not np.allclose(t, r)
