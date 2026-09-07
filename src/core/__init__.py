"""Núcleo sem UI: simulação, rede e treino. Depende só de numpy + stdlib."""

from .config import (
    Config,
    DQNCfg,
    FitnessCfg,
    GACfg,
    NetworkCfg,
    PhysicsCfg,
    SensorCfg,
    SimCfg,
)
from .dqn import STEER_ACTIONS, DQNTrainer, QNet
from .network import MLP
from .simulation import Activation, Frame, evaluate_population, simulate
from .track import CarState, Track
from .trainer import Trainer

__all__ = [
    "Config",
    "NetworkCfg",
    "SensorCfg",
    "PhysicsCfg",
    "GACfg",
    "FitnessCfg",
    "SimCfg",
    "DQNCfg",
    "MLP",
    "Track",
    "CarState",
    "Activation",
    "Frame",
    "simulate",
    "evaluate_population",
    "Trainer",
    "DQNTrainer",
    "QNet",
    "STEER_ACTIONS",
]
