"""Gera as pistas embutidas (JSON) a partir de curvas polares suaves.

Pode usar libs pesadas (roda offline). Uso:  python tools/make_track.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent / "src" / "tracks"


def build(name: str, radius_fn, width: float, n: int = 72) -> dict:
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r = radius_fn(th)
    center = np.column_stack([r * np.cos(th), r * np.sin(th)])
    tangent = center[1] - center[-1]
    heading = float(np.degrees(np.arctan2(tangent[1], tangent[0])))
    return {
        "name": name,
        "width_u": width,
        "closed": True,
        "centerline": [[round(float(x), 2), round(float(y), 2)] for x, y in center],
        "start": {
            "pos": [round(float(center[0, 0]), 2), round(float(center[0, 1]), 2)],
            "heading_deg": round(heading, 2),
        },
        "meta": {"author": "tools/make_track.py", "notes": name},
    }


# Amplitudes escolhidas para: (a) exigir direção reativa (nenhum controle fixo dá a volta)
# e (b) manter o raio de curvatura mínimo bem acima de metade da largura (offset sem cruzar).
TRACKS = [
    build("circuito_1", lambda t: 210 + 60 * np.sin(3 * t), width=48),
    build("circuito_2", lambda t: 210 + 48 * np.sin(2 * t) + 14 * np.sin(5 * t), width=44),
    build("circuito_3", lambda t: 215 + 40 * np.cos(4 * t) + 12 * np.sin(3 * t), width=42),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for tk in TRACKS:
        path = OUT / f"{tk['name']}.json"
        path.write_text(json.dumps(tk, indent=2), encoding="utf-8")
        print("escreveu", path.relative_to(OUT.parents[3]))


if __name__ == "__main__":
    main()
