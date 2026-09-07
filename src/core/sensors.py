"""Fachada fina para a leitura de sensores (o raycast vive em Track/geometry)."""

from __future__ import annotations

import numpy as np

from .config import SensorCfg
from .geometry import fan_angles
from .track import Track


def relative_angles(cfg: SensorCfg) -> np.ndarray:
    """Ângulos dos sensores relativos à frente do carro (rad)."""
    return fan_angles(cfg.count, cfg.fov_deg)


def read(track: Track, pos, heading: float, cfg: SensorCfg) -> np.ndarray:
    """Distâncias normalizadas [0, 1] até a parede (1 = nada dentro do alcance)."""
    return track.sensor_readings(pos, heading, cfg)
