"""Desenha a pista, o carro em foco e os raios dos sensores."""

from __future__ import annotations

import numpy as np
import pygame

from core.geometry import fan_angles

from . import theme


def draw(surf, cam, track, pos, heading, cfg):
    """Desenha a pista; se `pos`/`heading` forem dados, o carro e os raios de sensor."""
    pygame.draw.polygon(surf, theme.ROAD, cam.poly(track.outer))
    pygame.draw.polygon(surf, theme.BG, cam.poly(track.inner))

    center = track.center
    n = len(center)
    for i in range(0, n, 6):
        a = cam.to_screen(center[i])
        b = cam.to_screen(center[(i + 3) % n])
        pygame.draw.line(surf, theme.CENTERLINE, a, b, 1)

    # linha de largada
    s = track.start_state()
    fwd = np.array([np.cos(s.heading), np.sin(s.heading)])
    side = np.array([-fwd[1], fwd[0]]) * track.half_w
    pygame.draw.line(surf, theme.TEXT_DIM,
                     cam.to_screen(s.pos + side), cam.to_screen(s.pos - side), 2)

    if pos is None:
        return

    rel = fan_angles(cfg.sensors.count, cfg.sensors.fov_deg)
    readings = track.sensor_readings(pos, heading, cfg.sensors)
    for r, rd in zip(rel, readings):
        ang = heading + r
        end = pos + rd * cfg.sensors.range_u * np.array([np.cos(ang), np.sin(ang)])
        col = theme.SENSOR_FAR if rd > 0.33 else theme.SENSOR_NEAR
        pygame.draw.line(surf, col, cam.to_screen(pos), cam.to_screen(end), 1)
        pygame.draw.circle(surf, col, cam.to_screen(end), 3)

    _car(surf, cam, pos, heading)


# silhueta simples de carro visto de cima, em coordenadas locais (x = frente, y = lado).
_BODY = [(11, -4.5), (12.5, -2.5), (12.5, 2.5), (11, 4.5),
         (-11, 4.5), (-12, 2.5), (-12, -2.5), (-11, -4.5)]
_CABIN = [(1.5, -3.6), (5.5, -2.6), (5.5, 2.6), (1.5, 3.6), (-4.5, 3.2), (-4.5, -3.2)]
_WHEELS = [(7.5, 5.4), (7.5, -5.4), (-7.5, 5.4), (-7.5, -5.4)]  # centros
_WHEEL = [(2.6, 1.4), (2.6, -1.4), (-2.6, -1.4), (-2.6, 1.4)]


def _place(cam, pos, fwd, side, pts):
    return [cam.to_screen(pos + fwd * px + side * py) for px, py in pts]


def _car(surf, cam, pos, heading):
    fwd = np.array([np.cos(heading), np.sin(heading)])
    side = np.array([-fwd[1], fwd[0]])
    for cx, cy in _WHEELS:
        c = pos + fwd * cx + side * cy
        pygame.draw.polygon(surf, theme.WHEEL, _place(cam, c, fwd, side, _WHEEL))
    pygame.draw.polygon(surf, theme.CAR, _place(cam, pos, fwd, side, _BODY))
    pygame.draw.polygon(surf, theme.CAR_DARK, _place(cam, pos, fwd, side, _BODY), 1)
    pygame.draw.polygon(surf, theme.CABIN, _place(cam, pos, fwd, side, _CABIN))
