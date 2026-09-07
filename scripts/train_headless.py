"""Treina sem interface e salva o melhor genoma. Valida o núcleo ('prova de que aprende').

Uso:  python scripts/train_headless.py [--track circuito_1] [--gens 60] [--pop 40] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core import Config, Trainer
from core.config import GACfg, SimCfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="circuito_1")
    ap.add_argument("--gens", type=int, default=45)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="best_genome.json")
    args = ap.parse_args()

    cfg = Config(
        ga=GACfg(population=args.pop, generations=args.gens, seed=args.seed),
        sim=SimCfg(track=args.track),
    )
    tr = Trainer(cfg)
    tr.start()
    t0 = time.perf_counter()
    while tr.state == "running":
        tr.advance_generation()
        if tr.gen % 5 == 0 or tr.gen == 1:
            print(f"gen {tr.gen:3d}  best={tr.best_fitness:6.2f}  "
                  f"mean={tr.mean_fitness:6.2f}  best_laps={tr.best_laps}")

    dt = time.perf_counter() - t0
    Path(args.out).write_text(json.dumps(tr.save(), indent=2), encoding="utf-8")
    print(
        f"\nconcluido em {dt:.1f}s | fitness final {tr.best_fitness:.2f} | "
        f"{tr.best_laps} volta(s) | genoma salvo em {args.out}"
    )
    if tr.best_laps < 1:
        raise SystemExit("AVISO: melhor carro nao completou 1 volta; ajuste os hiperparametros.")


if __name__ == "__main__":
    main()
