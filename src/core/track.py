"""Pista: geometria, sensores (raycast), colisão e progresso ao longo da central."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry as geo
from .config import SensorCfg

RESAMPLE_N = 200
CRASH_CUSHION = 3.0  # o carro "sai da pista" quando seu centro passa este tanto da parede
_TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"


@dataclass
class CarState:
    pos: np.ndarray
    heading: float
    speed: float = 0.0
    alive: bool = True
    crashed: bool = False
    finished: bool = False
    progress: float = 0.0          # frações de volta percorridas (inclui voltas inteiras)
    idx: int = 0                   # índice do ponto mais próximo na central
    laps: int = 0
    steps: int = 0
    last_steer: float = 0.0


class Track:
    def __init__(self, name: str, centerline, width: float, start_pos, start_heading: float):
        self.name = name
        self.width = float(width)
        self.half_w = self.width / 2
        self.center = geo.resample_closed(centerline, RESAMPLE_N)

        a = geo.offset_closed(self.center, self.half_w)
        b = geo.offset_closed(self.center, -self.half_w)
        if geo.polygon_area(a) >= geo.polygon_area(b):
            self.outer, self.inner = a, b
        else:
            self.outer, self.inner = b, a

        # segmentos de parede (para o raycast dos sensores)
        self.wall_a = np.vstack([self.outer, self.inner])
        self.wall_b = np.vstack([np.roll(self.outer, -1, axis=0), np.roll(self.inner, -1, axis=0)])

        ring = np.vstack([self.center, self.center[:1]])
        seg = np.hypot(*np.diff(ring, axis=0).T)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

        self.start_pos = np.asarray(start_pos, float)
        self.start_heading = float(start_heading)
        self._fan_cache: dict[tuple[int, float], np.ndarray] = {}

    # ------------------------------------------------------------------ carga
    @classmethod
    def from_dict(cls, d: dict) -> Track:
        st = d.get("start", {})
        center = d["centerline"]
        return cls(
            d["name"],
            np.asarray(center, float),
            d["width_u"],
            st.get("pos", center[0]),
            np.radians(st.get("heading_deg", 0.0)),
        )

    @classmethod
    def load(cls, name: str) -> Track:
        with open(_TRACKS_DIR / f"{name}.json", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @staticmethod
    def list_names() -> list[str]:
        return sorted(p.stem for p in _TRACKS_DIR.glob("*.json"))

    # ------------------------------------------------------------------ uso
    @property
    def crash_dist(self) -> float:
        return self.half_w + CRASH_CUSHION

    def fan(self, cfg: SensorCfg) -> np.ndarray:
        key = (cfg.count, cfg.fov_deg)
        rel = self._fan_cache.get(key)
        if rel is None:
            rel = geo.fan_angles(cfg.count, cfg.fov_deg)
            self._fan_cache[key] = rel
        return rel

    def start_state(self) -> CarState:
        i = int(np.argmin(np.hypot(*(self.center - self.start_pos).T)))
        return CarState(pos=self.start_pos.copy(), heading=self.start_heading, idx=i)

    def sensor_readings(self, pos, heading: float, cfg: SensorCfg) -> np.ndarray:
        ang = heading + self.fan(cfg)
        dirs = np.column_stack([np.cos(ang), np.sin(ang)])
        d = geo.ray_fan_distances(pos, dirs, self.wall_a, self.wall_b, cfg.range_u)
        return d / cfg.range_u

    def locate(self, pos) -> tuple[int, float]:
        """Índice do ponto mais próximo na central e a distância até ele."""
        dx = self.center[:, 0] - pos[0]
        dy = self.center[:, 1] - pos[1]
        d2 = dx * dx + dy * dy
        i = int(d2.argmin())
        return i, float(np.sqrt(d2[i]))

    def commit_progress(self, s: CarState, i: int) -> None:
        n = len(self.center)
        delta = i - s.idx
        if delta < -n // 2:
            s.laps += 1
        elif delta > n // 2:
            s.laps = max(0, s.laps - 1)
        s.idx = i
        s.progress = s.laps + self.cum[i] / self.length
        if s.laps >= 1:
            s.finished = True
