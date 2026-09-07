"""Teste de aprendizado ponta a ponta (lento). Rode com:  pytest -m slow"""

import pytest

from core import Config, Trainer
from core.config import GACfg


@pytest.mark.slow
def test_car_learns_to_drive_a_lap():
    cfg = Config(ga=GACfg(seed=1))  # demais hiperparâmetros nos valores padrão
    tr = Trainer(cfg)
    tr.start()
    while tr.state == "running":
        tr.advance_generation()

    first_gen_best = tr.curve[0]
    assert tr.best_fitness > first_gen_best + 0.2, "o fitness deveria melhorar ao longo do treino"
    assert tr.best_laps >= 1, "o melhor carro deveria completar ao menos uma volta"
