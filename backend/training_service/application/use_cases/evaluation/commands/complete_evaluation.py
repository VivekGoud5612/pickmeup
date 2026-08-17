from __future__ import annotations

from backend.training_service.application.dto.evaluation.requests import (
    CompleteEvaluationRequest,
)
from backend.training_service.application.dto.evaluation.responses import (
    EvaluationSummaryResponse,
)
from backend.training_service.application.repositories.evaluation_repository import (
    EvaluationRepository,
)
from backend.training_service.application.mappers.evaluation_mapper import (
    EvaluationMapper,
)


class CompleteEvaluationUseCase:

    def __init__(
        self,
        evaluation_repo: EvaluationRepository,
    ) -> None:

        self._evaluation_repo = evaluation_repo

    def execute(
        self,
        request: CompleteEvaluationRequest,
    ) -> EvaluationSummaryResponse:

        evaluation = self._evaluation_repo.get_by_id(
            request.evaluation_id
        )

        evaluation.complete(request.metrics)

        self._evaluation_repo.update(evaluation)

        return EvaluationMapper.to_summary(evaluation)
