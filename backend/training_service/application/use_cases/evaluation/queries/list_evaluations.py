from __future__ import annotations

from training_service.application.dto.evaluation.requests import (
    ListEvaluationsRequest,
)
from training_service.application.dto.evaluation.responses import (
    ListEvaluationsResponse,
)
from training_service.application.repositories.evaluation_repository import (
    EvaluationRepository,
)
from training_service.application.mappers.evaluation_mapper import (
    EvaluationMapper,
)



class ListEvaluationsUseCase:

    def __init__(
        self,
        evaluation_repo: EvaluationRepository,
    ) -> None:

        self._evaluation_repo = evaluation_repo

    def execute(
        self,
        request: ListEvaluationsRequest,
    ) -> ListEvaluationsResponse:

        evaluations = self._evaluation_repo.list_all(
            request.checkpoint_id
        )

        return ListEvaluationsResponse(
            evaluations=[
                EvaluationMapper.to_summary(evaluation)
                for evaluation in evaluations
            ]
        )