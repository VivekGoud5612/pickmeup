from __future__ import annotations 
from dataclasses import dataclass 
from uuid import UUID 


@dataclass(slots=True, frozen=True, kw_only=True)
class EvaluationSummaryResponse:

    id: UUID

    checkpoint_id: UUID

    status: EvaluationStatus

    created_at: datetime

    finished_at: datetime | None


@dataclass(slots=True, frozen=True, kw_only=True)
class EvaluationCreatedResponse:

    evaluation: EvaluationSummaryResponse

    message: str = "Evaluation started successfully."


@dataclass(slots=True, frozen=True, kw_only=True)
class ListEvaluationsResponse:

    evaluations: list[EvaluationSummaryResponse]

@dataclass(slots = True, frozen = True, kw_only = True)
class DeleteEvaluationResponse:

    message : str = "Evaluation Deleted Successfully"