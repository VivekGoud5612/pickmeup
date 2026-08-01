from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from engine.utils.enums import EngineStatus

from backend.engine_gateway.application.dto.nested import (
    PerformanceMetrics,
)

from backend.training_service.domain.enums import (
    CheckpointType,
)



#Lifecycle
@dataclass(slots=True, frozen=True, kw_only=True)
class EngineInitializedResponse:
    status: EngineStatus

@dataclass(slots=True, frozen=True, kw_only=True)
class EngineTrainingStartedResponse:
    """
    Returned when the engine successfully starts training.
    """

    status: EngineStatus

    started_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineStatusResponse:
    """
    Returned after pause/resume/stop operations.
    """

    status: EngineStatus



#Metrics
@dataclass(slots=True, frozen=True, kw_only=True)
class EngineMetricsResponse:
    """
    Current training metrics from the engine.
    """

    metrics: PerformanceMetrics



#Checkpoint
@dataclass(slots=True, frozen=True, kw_only=True)
class EngineCheckpointSavedResponse:
    """
    Engine successfully saved a checkpoint.
    """

    checkpoint_path: Path

    checkpoint_type : CheckpointType = CheckpointType.MANUAL

    global_step : int

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineCheckpointEvaluationResponse:
    """
    Metrics obtained after evaluating a checkpoint.
    """

    metrics: PerformanceMetrics

    completed_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
