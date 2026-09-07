"""Network graph: nodes coloured by activation, edges highlighted by the signal
through them (weight x source activation)."""

from __future__ import annotations

import numpy as np
import pygame

from . import theme
from .i18n import t


def _columns(rect, sizes):
    pad = 30
    if len(sizes) == 1:
        xs = [rect.x + rect.w // 2]
    else:
        xs = [rect.x + pad + i * (rect.w - 2 * pad) // (len(sizes) - 1) for i in range(len(sizes))]
    cols = []
    for x, k in zip(xs, sizes):
        if k <= 1:
            ys = [rect.y + rect.h // 2]
        else:
            ys = [rect.y + pad + j * (rect.h - 2 * pad) // (k - 1) for j in range(k)]
        cols.append([(x, y) for y in ys])
    return cols


def draw(surf, rect, act, mlp, cfg, font, out_labels=None, caption=None):
    pygame.draw.rect(surf, theme.PANEL, rect, border_radius=6)
    pygame.draw.rect(surf, theme.PANEL_LINE, rect, 1, border_radius=6)

    values = [act.inputs]
    weights = [mlp.w1]
    if act.hidden.size:
        values.append(act.hidden)
        weights.append(mlp.w2)
    values.append(act.outputs)

    inner = rect.inflate(-10, -24).move(0, 8)
    cols = _columns(inner, [len(v) for v in values])

    # sinal por aresta: w[dst, src] * ativacao[src]
    signals = [w * values[k][None, :] for k, w in enumerate(weights)]
    scale = max(1e-6, max(float(np.abs(s).max()) for s in signals))

    for k, s in enumerate(signals):
        src_pts, dst_pts = cols[k], cols[k + 1]
        for j, pb in enumerate(dst_pts):
            for i, pa in enumerate(src_pts):
                inten = abs(float(s[j, i])) / scale
                if inten < 0.05:
                    pygame.draw.line(surf, theme.EDGE, pa, pb, 1)
                else:
                    base = theme.NODE_POS if s[j, i] >= 0 else theme.NODE_NEG
                    pygame.draw.line(surf, theme.lerp(theme.EDGE, base, inten), pa, pb,
                                     1 + int(round(2 * inten)))

    for vals, pts in zip(values, cols):
        for v, (x, y) in zip(vals, pts):
            pygame.draw.circle(surf, theme.diverging(v), (x, y), 9)
            pygame.draw.circle(surf, theme.PANEL_LINE, (x, y), 9, 1)

    in_labels = [f"S{i}" for i in range(cfg.sensors.count)] + [t("net.in.speed")]
    out_labels = out_labels or (t("net.out.steer"), t("net.out.accel"))
    for lbl, (x, y) in zip(in_labels, cols[0]):
        surf.blit(font.render(lbl, True, theme.TEXT_DIM), (x - 28, y - 6))
    for lbl, (x, y) in zip(out_labels, cols[-1]):
        surf.blit(font.render(lbl, True, theme.TEXT_DIM), (x + 13, y - 6))

    surf.blit(font.render(caption or t("net.caption"),
                          True, theme.TEXT_DIM), (rect.x + 10, rect.y + 6))
