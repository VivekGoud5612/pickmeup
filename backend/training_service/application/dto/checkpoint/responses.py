from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from backend.contracts.engine.enums import CheckpointType
from backend.training_service.domain.value_objects import TrainingProgress


@dataclass(slots=True, frozen=True, kw_only=True)
class CheckpointSummaryResponse:
    id: UUID
    training_run_id: UUID
    progress: TrainingProgress
    checkpoint_type: CheckpointType
    is_best: bool
    created_at: datetime


@dataclass(slots=True, frozen=True, kw_only=True)
class CheckpointCreatedResponse:
    checkpoint_summary: CheckpointSummaryResponse
    message: str = "Checkpoint saved successfully."


@dataclass(slots=True, frozen=True, kw_only=True)
class CheckpointDeletedResponse:
    message: str = "Checkpoint deleted successfully."


@dataclass(slots=True, frozen=True, kw_only=True)
class ListCheckpointsResponse:
    checkpoints_summary: list[CheckpointSummaryResponse]
