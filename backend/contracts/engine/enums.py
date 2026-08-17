from __future__ import annotations

from enum import Enum, IntEnum, auto


class TrainingAlgorithm(Enum):
    PPO = auto()
    MAPPO = auto()
    DQN = auto()
    SAC = auto()


class EvaluationStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class CheckpointType(Enum):
    """
    Useful later, when we want to specify how we save checkpoints.
    """

    MANUAL = auto()
    PERIODIC = auto()
    BEST_MODEL = auto()


class AgentRole(IntEnum):
    """Stable cross-service vocabulary for agent-specific measurements."""

    TANK = 0
    DEALER = 1
    HEALER = 2
    BOSS = 3


class EngineStatus(Enum):
    CREATED = auto()
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    FAILED = auto()
