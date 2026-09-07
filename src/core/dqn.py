"""Solução 2 — Deep Q-Network (DQN).

Alternativa **por gradiente** ao algoritmo genético. Um mesmo carro interage com a pista
passo a passo; suas transições `(estado, ação, recompensa, próximo estado)` entram num
*replay buffer*; a rede Q é ajustada por *backprop* para prever o retorno de cada ação.

- Estado  = leituras de sensor + velocidade normalizada (idêntico ao GA).
- Ações   = 5 discretas: virar {-1, -0.5, 0, +0.5, +1} com acelerador fixo.
- Recompensa por passo = ganho de progresso na pista (denso) − punição ao bater.

Sem PyTorch: a rede é uma MLP de 1 camada oculta (ReLU) em NumPy, com *backprop* manual e
otimizador Adam. Rede-alvo separada e ε-greedy com decaimento.
"""

from __future__ import annotations

import time

import numpy as np

from .config import Config, DQNCfg
from .simulation import step_car
from .track import Track

STEER_ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
THROTTLE = 0.85
STALL_STEPS = 300
R_PROGRESS = 60.0   # escala do ganho de progresso (uma volta ~ +60 de recompensa)
R_CRASH = 3.0


class QNet:
    """MLP 1 camada oculta (ReLU) -> Q linear. forward em lote + backprop + Adam."""

    def __init__(self, n_in: int, n_out: int, hidden: int, rng: np.random.Generator):
        self.n_in, self.n_out, self.hidden = int(n_in), int(n_out), int(hidden)
        self.w1 = rng.normal(0, np.sqrt(2 / n_in), (hidden, n_in)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.w2 = rng.normal(0, np.sqrt(2 / hidden), (n_out, hidden)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _params(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, x):
        single = np.ndim(x) == 1
        x = np.atleast_2d(np.asarray(x, np.float32))
        z1 = x @ self.w1.T + self.b1
        h = np.maximum(z1, 0.0)
        q = h @ self.w2.T + self.b2
        self._cache = (x, z1, h)
        return (q[0], h[0]) if single else (q, h)

    def train_step(self, states, actions, targets, lr: float) -> float:
        q, _ = self.forward(states)
        x, z1, h = self._cache
        n = len(states)
        pred = q[np.arange(n), actions]
        err = np.clip(pred - targets, -1.0, 1.0)      # gradiente limitado (estilo Huber)
        dq = np.zeros_like(q)
        dq[np.arange(n), actions] = err / n
        gw2 = dq.T @ h
        gb2 = dq.sum(0)
        dz1 = (dq @ self.w2) * (z1 > 0)
        gw1 = dz1.T @ x
        gb1 = dz1.sum(0)
        self._adam([gw1, gb1, gw2, gb2], lr)
        return float(np.mean(err ** 2))

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for p, g, m, v in zip(self._params(), grads, self._m, self._v):
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * (g * g)
            mhat = m / (1 - b1 ** self._t)
            vhat = v / (1 - b2 ** self._t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def copy_from(self, other: QNet) -> None:
        for p, q in zip(self._params(), other._params()):
            p[:] = q


class Replay:
    def __init__(self, cap: int, n_in: int, rng: np.random.Generator):
        self.cap, self.rng, self.n, self.i = int(cap), rng, 0, 0
        self.s = np.zeros((cap, n_in), np.float32)
        self.s2 = np.zeros((cap, n_in), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)

    def push(self, s, a, r, s2, done):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = self.rng.integers(0, self.n, size=b)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class DQNTrainer:
    """Mesma interface do `Trainer` do GA: state / step / reset / curve / best_*."""

    def __init__(self, cfg: Config | None = None, dqn: DQNCfg | None = None):
        self.cfg = cfg or Config()
        self.dqn = dqn or DQNCfg()
        self.reset()

    def reset(self, cfg: Config | None = None, dqn: DQNCfg | None = None) -> None:
        if cfg is not None:
            self.cfg = cfg
        if dqn is not None:
            self.dqn = dqn
        d = self.dqn
        self.rng = np.random.default_rng(d.seed)
        self.track = Track.load(self.cfg.sim.track)
        self.n_in = self.cfg.sensors.count + 1
        self.n_act = len(STEER_ACTIONS)
        self.q = QNet(self.n_in, self.n_act, d.hidden, self.rng)
        self.target = QNet(self.n_in, self.n_act, d.hidden, self.rng)
        self.target.copy_from(self.q)
        self.buf = Replay(d.buffer, self.n_in, self.rng)

        self.state = "idle"
        self.steps = 0
        self.episodes = 0
        self.eps = d.eps_start
        self.loss = 0.0
        self.curve: list[float] = []       # recompensa por episódio
        self.best_reward = -1e9
        self.best_laps = 0

        # "carro-demonstração": roda a política gulosa atual a ritmo fixo (1 passo/frame),
        # desacoplado da velocidade do treino — é o que a tela desenha.
        self.demo_q = np.zeros(self.n_act)
        self.demo_obs = np.zeros(self.n_in)
        self.demo_hidden = np.zeros(d.hidden)
        self.demo_pos = self.track.start_pos.copy()
        self.demo_heading = self.track.start_heading
        self._begin_episode()
        self._demo_reset()

    # ------------------------------------------------------------------ episódio
    def _begin_episode(self) -> None:
        self._car = self.track.start_state()
        self._obs = self._obs_of(self._car)
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._last_gain = 0

    def _obs_of(self, car) -> np.ndarray:
        r = self.track.sensor_readings(car.pos, car.heading, self.cfg.sensors)
        return np.concatenate([r, [car.speed / self.cfg.physics.v_max]]).astype(np.float32)

    # ------------------------------------------------------------------ demonstração
    def _demo_reset(self) -> None:
        self._demo = self.track.start_state()
        self._demo_obs = self._obs_of(self._demo)
        self._demo_steps = 0
        self._demo_last_gain = 0
        self.demo_pos = self._demo.pos.copy()
        self.demo_heading = self._demo.heading

    def demo_step(self) -> None:
        """Avança 1 passo o carro-demonstração com a política gulosa (para a visualização)."""
        q, h = self.q.forward(self._demo_obs)
        a = int(np.argmax(q))
        prev = self._demo.progress
        step_car(self._demo, float(STEER_ACTIONS[a]), THROTTLE, self.cfg.physics)
        self._demo_steps += 1
        i, dist = self.track.locate(self._demo.pos)
        crashed = dist > self.track.crash_dist
        if not crashed:
            self.track.commit_progress(self._demo, i)
        if self._demo.progress > prev + 1e-4:
            self._demo_last_gain = self._demo_steps
        stalled = self._demo_steps - self._demo_last_gain > STALL_STEPS
        if crashed or stalled:
            self._demo_reset()
            return
        self._demo_obs = self._obs_of(self._demo)
        self.demo_pos = self._demo.pos.copy()
        self.demo_heading = self._demo.heading
        self.demo_q, self.demo_hidden, self.demo_obs = q, h, self._demo_obs

    # ------------------------------------------------------------------ controle
    def start(self):
        if self.state != "done":
            self.state = "running"

    def pause(self):
        if self.state == "running":
            self.state = "idle"

    def toggle(self):
        self.pause() if self.state == "running" else self.start()

    # ------------------------------------------------------------------ treino
    def step(self, budget_ms: float = 8.0) -> None:
        if self.state != "running":
            return
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) * 1000.0 < budget_ms:
            self._env_step()
            if self.buf.n >= self.dqn.warmup:
                self._learn()

    def _epsilon(self) -> float:
        d = self.dqn
        frac = min(1.0, self.steps / max(1, d.eps_decay_steps))
        return d.eps_start + frac * (d.eps_end - d.eps_start)

    def _env_step(self) -> None:
        d = self.dqn
        self.eps = self._epsilon()
        q, h = self.q.forward(self._obs)

        if self.rng.random() < self.eps:
            a = int(self.rng.integers(0, self.n_act))
        else:
            a = int(np.argmax(q))

        prev_progress = self._car.progress
        step_car(self._car, float(STEER_ACTIONS[a]), THROTTLE, self.cfg.physics)
        self._ep_steps += 1
        self.steps += 1

        i, dist = self.track.locate(self._car.pos)
        crashed = dist > self.track.crash_dist
        if not crashed:
            self.track.commit_progress(self._car, i)

        gain = self._car.progress - prev_progress
        if gain > 1e-4:
            self._last_gain = self._ep_steps
        stalled = self._ep_steps - self._last_gain > STALL_STEPS
        done = crashed or stalled or self._ep_steps >= d.max_steps

        reward = R_PROGRESS * gain
        if crashed:
            reward -= R_CRASH

        obs2 = self._obs_of(self._car)
        self.buf.push(self._obs, a, reward, obs2, done)
        self._obs = obs2
        self._ep_reward += reward

        if done:
            self.episodes += 1
            self.curve.append(self._ep_reward)
            if self._ep_reward > self.best_reward:
                self.best_reward = self._ep_reward
                self.best_laps = int(self._car.laps)
            self._begin_episode()

    def _learn(self) -> None:
        d = self.dqn
        s, a, r, s2, done = self.buf.sample(d.batch)
        q2, _ = self.target.forward(s2)
        target = r + d.gamma * np.max(q2, axis=1) * (1.0 - done)
        self.loss = self.q.train_step(s, a, target, d.lr)
        if self.steps % d.target_sync == 0:
            self.target.copy_from(self.q)

    # ------------------------------------------------------------------ persistência
    def save(self) -> dict:
        return {
            "kind": "dqn",
            "config": self.cfg.to_dict(),
            "dqn": vars(self.dqn) if not hasattr(self.dqn, "__dataclass_fields__")
            else {f: getattr(self.dqn, f) for f in self.dqn.__dataclass_fields__},
            "weights": [p.tolist() for p in self.q._params()],
            "episodes": self.episodes,
            "best_reward": self.best_reward,
        }
