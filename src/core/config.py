"""Hiperparâmetros do projeto, como dataclasses imutáveis + (de)serialização JSON."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NetworkCfg:
    hidden: int = 6              # 0 = rede linear (sem camada oculta)
    activation: str = "tanh"     # "tanh" | "relu"


@dataclass(frozen=True)
class SensorCfg:
    count: int = 5
    fov_deg: float = 160.0
    range_u: float = 150.0


@dataclass(frozen=True)
class PhysicsCfg:
    v_max: float = 140.0
    accel: float = 120.0
    steer_deg_s: float = 170.0
    friction: float = 0.02
    dt: float = 1.0 / 60.0


@dataclass(frozen=True)
class GACfg:
    population: int = 30
    generations: int = 45
    mutation_rate: float = 0.12
    mutation_sigma: float = 0.18
    elitism: int = 2
    tournament_k: int = 3
    crossover: bool = True
    seed: int = 0


@dataclass(frozen=True)
class FitnessCfg:
    w_progress: float = 1.0      # distância ao longo da central (conta voltas inteiras)
    w_speed: float = 0.4         # incentiva andar rápido, não só sobreviver
    w_smooth: float = 0.05       # penaliza zigue-zague
    crash_penalty: float = 1.0
    lap_bonus: float = 0.5       # empurrãozinho ao cruzar a linha (1ª volta visível na curva)


@dataclass(frozen=True)
class SimCfg:
    max_steps: int = 1300      # ~1 volta a boa velocidade; mantém o treino ágil (didático)
    track: str = "circuito_1"


@dataclass(frozen=True)
class DQNCfg:
    """Hiperparâmetros da 2ª solução (Deep Q-Network). Usa os mesmos sensores/física."""
    hidden: int = 32
    lr: float = 1e-3
    gamma: float = 0.99
    batch: int = 64
    buffer: int = 20_000
    warmup: int = 800          # passos antes de começar a aprender
    target_sync: int = 400     # a cada N passos, copia a rede online -> alvo
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 12_000
    max_steps: int = 1300      # limite de passos por episódio
    seed: int = 0


@dataclass(frozen=True)
class Config:
    network: NetworkCfg = NetworkCfg()
    sensors: SensorCfg = SensorCfg()
    physics: PhysicsCfg = PhysicsCfg()
    ga: GACfg = GACfg()
    fitness: FitnessCfg = FitnessCfg()
    sim: SimCfg = SimCfg()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(
            network=NetworkCfg(**d.get("network", {})),
            sensors=SensorCfg(**d.get("sensors", {})),
            physics=PhysicsCfg(**d.get("physics", {})),
            ga=GACfg(**d.get("ga", {})),
            fitness=FitnessCfg(**d.get("fitness", {})),
            sim=SimCfg(**d.get("sim", {})),
        )
