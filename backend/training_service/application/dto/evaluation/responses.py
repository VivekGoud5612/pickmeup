from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from backend.training_service.domain.enums import EvaluationStatus
from backend.contracts.engine.models.performance_metrics import PerformanceMetrics


@dataclass(slots=True, frozen=True, kw_only=True)
class EvaluationSummaryResponse:
    id: UUID
    checkpoint_id: UUID
    status: EvaluationStatus
    metrics: PerformanceMetrics | None
    created_at: datetime
    finished_at: datetime | None


@dataclass(slots=True, frozen=True, kw_only=True)
class EvaluationCreatedResponse:
    evaluation_summary: EvaluationSummaryResponse
    message: str = "Evaluation started successfully."


@dataclass(slots=True, frozen=True, kw_only=True)
class ListEvaluationsResponse:
    evaluations_summary: list[EvaluationSummaryResponse]


@dataclass(slots=True, frozen=True, kw_only=True)
class EvaluationDeletedResponse:
    message: str = "Evaluation Deleted Successfully"
