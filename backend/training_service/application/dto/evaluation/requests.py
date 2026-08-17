from __future__ import annotations 
from dataclasses import dataclass 
from uuid import UUID 

from backend.contracts.engine.models.performance_metrics import PerformanceMetrics


@dataclass(slots=True, frozen=True, kw_only=True)
class StartEvaluationRequest:

    checkpoint_id: UUID

    evaluation_episodes : int 

    notes: str = ""


@dataclass(slots=True, frozen=True, kw_only=True)
class CompleteEvaluationRequest:

    evaluation_id: UUID
    metrics: PerformanceMetrics

@dataclass(slots=True, frozen=True, kw_only=True)
class FailEvaluationRequest:

    evaluation_id: UUID

@dataclass(slots=True, frozen=True, kw_only=True)
class GetEvaluationRequest:

    evaluation_id: UUID

@dataclass(slots=True, frozen=True, kw_only=True)
class ListEvaluationsRequest:

    checkpoint_id: UUID

@dataclass(slots = True, frozen = True, kw_only = True)
class DeleteEvaluationRequest:
    evaluation_id: UUID
