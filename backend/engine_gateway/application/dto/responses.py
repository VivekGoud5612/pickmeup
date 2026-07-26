from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from uuid import UUID
from pydantic import field

from training_service.domain.enums import TrainingStatus

from training_service.domain.value_objects import (
    PerformanceMetrics,
    TrainingProgress,
)
from engine.utils.enums import EngineStatus 

from pathlib import Path

from engine_gateway.application.dto.nested_requests import PerformanceMetrics

from training_service.domain.enums import CheckpointType


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineTrainingStartedResponse:

    status: EngineStatus

    started_at: datetime = field(default_factory = lambda : datetime.now(UTC))


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineStatusResponse:

    status: EngineStatus 


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineMetricsResponse:
    """
    Snapshot returned by the Engine.
    """

    progress: TrainingProgress   ## Need to check what this is as well

    metrics: PerformanceMetrics


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineCheckpointSavedResponse:

    checkpoint_path: Path

    checkpoint_type : CheckpointType

    created_at: datetime = field(default_factory = lambda : datetime.now(UTC))

@dataclass(slots=True, frozen=True, kw_only=True)
class EngineCheckpointEvaluationResponse:

    checkpoint_path : Path   ## Lets think if this is necessary

    metrics: PerformanceMetrics

    completed_at: datetime = field(default_factory = lambda : datetime.now(UTC))


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineHeartbeatResponse:
    """
    Lightweight runtime information returned
    while training is active.
    """

    progress: TrainingProgress

    status: TrainingStatus
