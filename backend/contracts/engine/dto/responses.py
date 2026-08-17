from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from backend.contracts.engine.enums import CheckpointType
from backend.contracts.engine.models.performance_metrics import PerformanceMetrics
from backend.contracts.engine.models.training_progress import TrainingProgress
from backend.contracts.engine.enums import EngineStatus


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineTrainingStartedResponse:
    run_id : UUID
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
    metrics: PerformanceMetrics | None


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
    status: EngineStatus
