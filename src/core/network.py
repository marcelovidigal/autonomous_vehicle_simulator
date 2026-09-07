"""MLP rasa em NumPy. Pesos como vetor plano (para o GA) + snapshot de ativações."""

from __future__ import annotations

import numpy as np

from .config import NetworkCfg

_N_OUT = 2  # direção, acelerador


def _activation(name: str):
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    return np.tanh


class MLP:
    def __init__(self, n_in: int, cfg: NetworkCfg, rng: np.random.Generator | None = None):
        self.n_in = int(n_in)
        self.hidden = int(cfg.hidden)
        self.n_out = _N_OUT
        self._act = _activation(cfg.activation)
        self.activation = cfg.activation
        rng = rng or np.random.default_rng(0)

        def he(rows: int, fan_in: int) -> np.ndarray:
            return rng.normal(0.0, 1.0 / np.sqrt(fan_in), (rows, fan_in)).astype(np.float32)

        if self.hidden > 0:
            self.w1 = he(self.hidden, self.n_in)
            self.b1 = np.zeros(self.hidden, np.float32)
            self.w2 = he(self.n_out, self.hidden)
            self.b2 = np.zeros(self.n_out, np.float32)
        else:
            self.w1 = he(self.n_out, self.n_in)
            self.b1 = np.zeros(self.n_out, np.float32)
            self.w2 = None
            self.b2 = None
        self.last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @staticmethod
    def n_params(n_in: int, cfg: NetworkCfg) -> int:
        h = int(cfg.hidden)
        if h > 0:
            return h * n_in + h + _N_OUT * h + _N_OUT
        return _N_OUT * n_in + _N_OUT

    def forward(self, x) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.hidden > 0:
            h = self._act(self.w1 @ x + self.b1)
            out = np.tanh(self.w2 @ h + self.b2)
        else:
            h = np.zeros(0, np.float32)
            out = np.tanh(self.w1 @ x + self.b1)
        self.last = (x, h, out)
        return out

    def to_flat(self) -> np.ndarray:
        parts = [self.w1.ravel(), self.b1]
        if self.hidden > 0:
            parts += [self.w2.ravel(), self.b2]
        return np.concatenate(parts).astype(np.float32)

    def load_flat(self, vec) -> None:
        vec = np.asarray(vec, np.float32)
        i = 0

        def take(n: int) -> np.ndarray:
            nonlocal i
            chunk = vec[i : i + n]
            i += n
            return chunk

        if self.hidden > 0:
            self.w1 = take(self.hidden * self.n_in).reshape(self.hidden, self.n_in).copy()
            self.b1 = take(self.hidden).copy()
            self.w2 = take(self.n_out * self.hidden).reshape(self.n_out, self.hidden).copy()
            self.b2 = take(self.n_out).copy()
        else:
            self.w1 = take(self.n_out * self.n_in).reshape(self.n_out, self.n_in).copy()
            self.b1 = take(self.n_out).copy()

    @classmethod
    def from_flat(cls, vec, n_in: int, cfg: NetworkCfg) -> MLP:
        m = cls(n_in, cfg)
        m.load_flat(vec)
        return m
