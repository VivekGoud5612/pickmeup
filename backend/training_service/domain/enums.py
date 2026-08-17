from __future__ import annotations

from enum import Enum, auto

from backend.contracts.engine.enums import (
    CheckpointType,
    EvaluationStatus,
    TrainingAlgorithm,
)


class TrainingStatus(Enum):
    CREATED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    FAILED = auto()

__all__ = [
    "TrainingStatus",
    "TrainingAlgorithm",
    "EvaluationStatus",
    "CheckpointType",
]
