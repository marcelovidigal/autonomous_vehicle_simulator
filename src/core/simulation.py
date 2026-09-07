"""Roda genomas na pista e devolve o fitness. Determinístico (sem RNG).

`simulate`            -> um genoma, versão legível, opcionalmente grava o trace.
`simulate_population` -> toda a população de uma vez, vetorizada (usada no treino).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, PhysicsCfg
from .network import MLP
from .track import CarState, Track


@dataclass
class Activation:
    inputs: np.ndarray
    hidden: np.ndarray
    outputs: np.ndarray


@dataclass
class Frame:
    pos: np.ndarray
    heading: float
    act: Activation
    laps: int


@dataclass
class SimResult:
    fitness: float
    state: CarState
    trace: list[Frame] | None


STALL_STEPS = 300  # carro sem ganhar progresso por este tempo é encerrado (parado/girando)


def n_inputs(cfg: Config) -> int:
    return cfg.sensors.count + 1  # sensores + velocidade normalizada


def step_car(s: CarState, steer: float, throttle: float, phys: PhysicsCfg) -> None:
    """Modelo cinemático simples. `steer` e `throttle` vêm em [-1, 1]."""
    grip = float(np.clip(s.speed / (0.25 * phys.v_max), 0.0, 1.0))  # não vira parado
    s.heading += float(steer) * np.radians(phys.steer_deg_s) * phys.dt * grip
    s.speed += float(throttle) * phys.accel * phys.dt
    s.speed -= s.speed * phys.friction
    s.speed = float(np.clip(s.speed, 0.0, phys.v_max))
    s.pos = s.pos + s.speed * phys.dt * np.array([np.cos(s.heading), np.sin(s.heading)])


def _score(w, max_prog, speed_sum, steer_change, steps, crashed, laps):
    steps = np.maximum(1, steps)
    return (
        w.w_progress * max_prog
        + w.w_speed * (speed_sum / steps)
        - w.w_smooth * (steer_change / steps)
        - w.crash_penalty * np.asarray(crashed, float)
        + w.lap_bonus * laps
    )


def simulate(genome, track: Track, cfg: Config, record: bool = False) -> SimResult:
    mlp = MLP.from_flat(genome, n_inputs(cfg), cfg.network)
    s = track.start_state()
    phys, fit = cfg.physics, cfg.fitness
    trace: list[Frame] | None = [] if record else None
    speed_sum = steer_change = max_progress = 0.0
    last_gain_step = 0

    for _ in range(cfg.sim.max_steps):
        readings = track.sensor_readings(s.pos, s.heading, cfg.sensors)
        x = np.concatenate([readings, [s.speed / phys.v_max]])
        out = mlp.forward(x)
        steer, throttle = float(out[0]), float(out[1])

        steer_change += abs(steer - s.last_steer)
        s.last_steer = steer
        step_car(s, steer, throttle, phys)
        s.steps += 1

        if record:
            inp, hid, o = mlp.last
            trace.append(Frame(s.pos.copy(), s.heading,
                               Activation(inp.copy(), hid.copy(), o.copy()), s.laps))

        i, dist = track.locate(s.pos)
        if dist > track.crash_dist:
            s.alive, s.crashed = False, True
            break
        track.commit_progress(s, i)
        if s.progress > max_progress + 1e-4:
            max_progress = s.progress
            last_gain_step = s.steps
        speed_sum += s.speed / phys.v_max
        if s.steps - last_gain_step > STALL_STEPS:  # parado/girando: encerra (várias voltas são OK)
            break

    score = float(_score(fit, max_progress, speed_sum, steer_change, s.steps, s.crashed, s.laps))
    return SimResult(score, s, trace)


def _unpack_batched(genomes: np.ndarray, n_in: int, hidden: int):
    """Fatia o vetor plano de cada genoma nos tensores de peso em lote (eixo 0 = população)."""
    p = len(genomes)
    g = np.asarray(genomes, np.float32)
    if hidden > 0:
        cuts = np.cumsum([hidden * n_in, hidden, 2 * hidden, 2])
        w1, b1, w2, b2 = np.split(g, cuts[:-1], axis=1)
        return w1.reshape(p, hidden, n_in), b1, w2.reshape(p, 2, hidden), b2
    w1, b1 = np.split(g, [2 * n_in], axis=1)
    return w1.reshape(p, 2, n_in), b1, None, None


def simulate_population(genomes, track: Track, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Avalia todos os genomas em lockstep. Retorna (fitness (P,), laps (P,))."""
    p = len(genomes)
    net, phys, fit = cfg.network, cfg.physics, cfg.fitness
    n_in = n_inputs(cfg)
    hidden = int(net.hidden)
    w1, b1, w2, b2 = _unpack_batched(genomes, n_in, hidden)
    act = (lambda z: np.maximum(z, 0.0)) if net.activation == "relu" else np.tanh

    st = track.start_state()
    pos = np.tile(st.pos.astype(float), (p, 1))
    heading = np.full(p, float(st.heading))
    speed = np.zeros(p)
    idx = np.full(p, st.idx, dtype=int)
    laps = np.zeros(p, dtype=int)
    last_steer = np.zeros(p)
    alive = np.ones(p, dtype=bool)
    crashed = np.zeros(p, dtype=bool)
    speed_sum = np.zeros(p)
    steer_change = np.zeros(p)
    max_prog = np.zeros(p)
    steps = np.zeros(p, dtype=int)
    last_gain = np.zeros(p, dtype=int)

    rel = track.fan(cfg.sensors)                       # (M,)
    seg_a, seg_b = track.wall_a, track.wall_b          # (N, 2)
    edge = seg_b - seg_a                               # (N, 2)
    center = track.center                              # (Nc, 2)
    n_c = len(center)
    steer_rate = np.radians(phys.steer_deg_s) * phys.dt
    vmax = phys.v_max
    rng_u = cfg.sensors.range_u
    crash_d = track.crash_dist

    for _ in range(cfg.sim.max_steps):
        m = alive
        if not m.any():
            break
        a = int(m.sum())

        # --- sensores (raycast vetorizado) para os carros vivos
        ang = heading[m][:, None] + rel[None, :]                       # (a, M)
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)           # (a, M, 2)
        perp = np.stack([-dirs[..., 1], dirs[..., 0]], axis=-1)        # (a, M, 2)
        v1 = pos[m][:, None, :] - seg_a[None, :, :]                    # (a, N, 2)
        denom = np.einsum("amk,nk->amn", perp, edge)                  # (a, M, N)
        cross = edge[:, 0] * v1[..., 1] - edge[:, 1] * v1[..., 0]      # (a, N)
        v1p = np.einsum("amk,ank->amn", perp, v1)                      # (a, M, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ray = cross[:, None, :] / denom
            t_seg = v1p / denom
        ok = (denom != 0) & (t_ray >= 0) & (t_seg >= 0) & (t_seg <= 1)
        t_ray = np.where(ok, t_ray, np.inf)
        readings = np.minimum(t_ray.min(axis=2), rng_u) / rng_u        # (a, M)

        # --- forward da MLP (em lote)
        x = np.concatenate([readings, (speed[m] / vmax)[:, None]], axis=1)
        if hidden > 0:
            h = act(np.einsum("ahi,ai->ah", w1[m], x) + b1[m])
            out = np.tanh(np.einsum("aoh,ah->ao", w2[m], h) + b2[m])
        else:
            out = np.tanh(np.einsum("aoi,ai->ao", w1[m], x) + b1[m])
        steer, throttle = out[:, 0], out[:, 1]

        # --- integração
        grip = np.clip(speed[m] / (0.25 * vmax), 0.0, 1.0)
        steer_change[m] += np.abs(steer - last_steer[m])
        last_steer[m] = steer
        heading[m] += steer * steer_rate * grip
        sp = speed[m] + throttle * phys.accel * phys.dt
        sp = np.clip(sp - sp * phys.friction, 0.0, vmax)
        speed[m] = sp
        pos[m] += (sp * phys.dt)[:, None] * np.stack(
            [np.cos(heading[m]), np.sin(heading[m])], axis=1
        )
        steps[m] += 1

        # --- colisão + progresso (distância à central)
        pm = pos[m]
        d2 = (center[None, :, 0] - pm[:, 0][:, None]) ** 2 + (
            center[None, :, 1] - pm[:, 1][:, None]
        ) ** 2
        ni = d2.argmin(axis=1)
        ndist = np.sqrt(d2[np.arange(a), ni])

        prev = idx[m]
        delta = ni - prev
        lm = laps[m]
        lm = np.where(delta < -n_c // 2, lm + 1, lm)
        lm = np.where(delta > n_c // 2, np.maximum(0, lm - 1), lm)
        laps[m] = lm
        idx[m] = ni

        prog = lm + track.cum[ni] / track.length
        gained = prog > max_prog[m] + 1e-4
        last_gain[m] = np.where(gained, steps[m], last_gain[m])
        max_prog[m] = np.maximum(max_prog[m], prog)
        speed_sum[m] += sp / vmax

        crash_now = ndist > crash_d
        stall_now = (steps[m] - last_gain[m]) > STALL_STEPS
        done_now = crash_now | stall_now
        cm = crashed[m]
        crashed[m] = cm | crash_now
        am = alive[m]
        alive[m] = am & ~done_now

    score = _score(fit, max_prog, speed_sum, steer_change, steps, crashed, laps)
    return score.astype(float), laps.astype(int)


def evaluate_population(genomes, track: Track, cfg: Config):
    """Versão legível (loop Python) — usada em testes e inspeção."""
    fitness = np.empty(len(genomes), float)
    states: list[CarState] = []
    for i, g in enumerate(genomes):
        r = simulate(g, track, cfg)
        fitness[i] = r.fitness
        states.append(r.state)
    return fitness, states
