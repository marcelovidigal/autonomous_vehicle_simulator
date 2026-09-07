"""Paleta e helper de cor divergente (ativações)."""

from __future__ import annotations

BG = (22, 24, 30)
PANEL = (30, 33, 41)
PANEL_LINE = (52, 57, 70)
ROAD = (44, 48, 58)
CENTERLINE = (74, 80, 96)
TEXT = (208, 213, 222)
TEXT_DIM = (150, 157, 170)
ACCENT = (120, 180, 255)
CAR = (255, 214, 90)
CAR_DARK = (150, 120, 40)
CABIN = (40, 46, 66)
WHEEL = (24, 24, 28)
SENSOR_FAR = (90, 200, 120)
SENSOR_NEAR = (232, 120, 96)
NODE_ZERO = (64, 68, 82)
NODE_POS = (232, 96, 84)
NODE_NEG = (72, 132, 232)
EDGE = (48, 52, 64)


def lerp(a, b, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def diverging(v: float) -> tuple[int, int, int]:
    """v em [-1, 1] -> cor (negativo azul, zero cinza, positivo vermelho)."""
    v = max(-1.0, min(1.0, float(v)))
    return lerp(NODE_ZERO, NODE_POS if v >= 0 else NODE_NEG, abs(v))
