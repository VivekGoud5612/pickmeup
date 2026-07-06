from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from training_service.domain.enums import TrainingStatus

from training_service.domain.value_objects import (
    PerformanceMetrics,
    TrainingProgress,
)

from pathlib import Path

from engine_gateway.application.dto.nested_requests import PerformanceMetrics


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineTrainingStartedResponse:

    training_run_id: UUID

    status: TrainingStatus

    started_at: datetime


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineStatusResponse:

    training_run_id: UUID

    status: TrainingStatus


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineMetricsResponse:
    """
    Snapshot returned by the Engine.
    """

    training_run_id: UUID

    progress: TrainingProgress

    metrics: PerformanceMetrics


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineCheckpointResponse:

    checkpoint_id: UUID

    checkpoint_path: Path

    created_at: datetime


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineEvaluationResponse:

    checkpoint_id: UUID

    metrics: PerformanceMetrics

    completed_at: datetime


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineHeartbeatResponse:
    """
    Lightweight runtime information returned
    while training is active.
    """

    training_run_id: UUID

    progress: TrainingProgress

    status: TrainingStatus