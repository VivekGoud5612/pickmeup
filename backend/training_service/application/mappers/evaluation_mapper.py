from __future__ import annotations

from training_service.domain.entities.evaluation import (
    EvaluationResult,
)

from training_service.application.dto.evaluation.responses import (
    EvaluationSummaryResponse,
    EvaluationCreatedResponse,
    DeleteEvaluationResponse,
)


class EvaluationMapper:
    """
    Maps evaluation entities into response DTOs.
    """

    @staticmethod
    def to_summary(
        evaluation: EvaluationResult,
    ) -> EvaluationSummaryResponse:

        return EvaluationSummaryResponse(
            id=evaluation.id,
            checkpoint_id=evaluation.checkpoint_id,
            status=evaluation.status,
            metrics=evaluation.metrics,
            created_at=evaluation.created_at,
            finished_at=evaluation.finished_at,
        )

    @staticmethod
    def to_created(
        evaluation: EvaluationResult,
    ) -> EvaluationCreatedResponse:

        return EvaluationCreatedResponse(
            summary=EvaluationMapper.to_summary(
                evaluation
            ),
        )


    @staticmethod 
    def to_deleted(
        evaluation : EvaluationResult,
    ) -> DeleteEvaluationResponse:

        return DeleteEvaluationResponse(
            message=f"Checkpoint '{evaluation.id}' deleted successfully."
        )