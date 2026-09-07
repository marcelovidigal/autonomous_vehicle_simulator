"""Geometria 2D em NumPy puro (compatível com Pyodide/WASM — sem shapely)."""

from __future__ import annotations

import numpy as np


def polyline_length(pts, closed: bool = True) -> float:
    p = np.asarray(pts, float)
    d = np.diff(p, axis=0)
    total = float(np.hypot(d[:, 0], d[:, 1]).sum())
    if closed:
        total += float(np.hypot(*(p[0] - p[-1])))
    return total


def resample_closed(pts, n: int) -> np.ndarray:
    """Reamostra uma polilinha fechada em `n` pontos igualmente espaçados por arco."""
    p = np.asarray(pts, float)
    p = np.vstack([p, p[:1]])
    d = np.diff(p, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, s[-1], n, endpoint=False)
    x = np.interp(targets, s, p[:, 0])
    y = np.interp(targets, s, p[:, 1])
    return np.column_stack([x, y])


def _vertex_normals_closed(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, float)
    tang = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    n = np.column_stack([-tang[:, 1], tang[:, 0]])
    ln = np.hypot(n[:, 0], n[:, 1])
    ln[ln == 0] = 1.0
    return n / ln[:, None]


def offset_closed(pts, dist: float) -> np.ndarray:
    """Desloca uma polilinha fechada ao longo da normal do vértice por `dist`."""
    p = np.asarray(pts, float)
    return p + _vertex_normals_closed(p) * dist


def polygon_area(poly) -> float:
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_polygon(pt, poly) -> bool:
    x, y = float(pt[0]), float(pt[1])
    p = np.asarray(poly, float)
    x1, y1 = p[:, 0], p[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    straddle = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.sum(straddle & (x < x_cross)) & 1)


def fan_angles(count: int, fov_deg: float) -> np.ndarray:
    """Ângulos relativos (rad) de um leque de `count` sensores, centrado em 0 (frente)."""
    if count <= 1:
        return np.array([0.0])
    half = np.radians(fov_deg) / 2.0
    return np.linspace(-half, half, count)


def ray_fan_distances(origin, dirs, seg_a, seg_b, max_dist: float) -> np.ndarray:
    """Distância do 1º toque de cada raio (direção unitária em `dirs`, M raios) contra
    os segmentos seg_a[i] -> seg_b[i] (N segmentos). Retorna (M,) já limitado a max_dist.
    """
    o = np.asarray(origin, float)
    dirs = np.asarray(dirs, float)                       # (M, 2)
    a = np.asarray(seg_a, float)                         # (N, 2)
    b = np.asarray(seg_b, float)                         # (N, 2)
    v1 = o - a                                           # (N, 2)
    v2 = b - a                                           # (N, 2)
    perp = np.column_stack([-dirs[:, 1], dirs[:, 0]])    # (M, 2)
    denom = perp @ v2.T                                  # (M, N)
    cross = v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]    # (N,)
    v1p = perp @ v1.T                                    # (M, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ray = cross[None, :] / denom                   # distância ao longo do raio
        t_seg = v1p / denom                              # parâmetro no segmento [0, 1]
    hit = (denom != 0) & (t_ray >= 0) & (t_seg >= 0) & (t_seg <= 1)
    t_ray = np.where(hit, t_ray, np.inf)
    return np.minimum(t_ray.min(axis=1), max_dist)
