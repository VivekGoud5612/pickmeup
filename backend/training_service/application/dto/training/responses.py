"""
Response DTOs returned by Training use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from backend.training_service.domain.enums import TrainingStatus
from backend.training_service.domain.value_objects import TrainingProgress


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingSummaryResponse:
    id: UUID
    name: str
    status: TrainingStatus
    progress: TrainingProgress | None
    started_at: datetime
    finished_at: datetime | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingCreatedResponse:
    run_summary: TrainingSummaryResponse
    message: str = "Training Created Successfully"


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingProgressResponse:
    id: UUID
    name: str
    status: TrainingStatus
    progress: TrainingProgress | None


@dataclass(slots=True, frozen=True, kw_only=True)
class ListTrainingRunsResponse:
    runs_summary: list[TrainingSummaryResponse]


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingDeletedResponse:
    message: str = "Training deleted Successfully"
