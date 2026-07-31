from __future__ import annotations

from backend.training_service.application.dto.evaluation.requests import (
    DeleteEvaluationRequest,
)
from backend.training_service.application.dto.evaluation.responses import (
    EvaluationDeletedResponse,
)
from backend.training_service.application.repositories.evaluation_repository import (
    EvaluationRepository,
)
from backend.training_service.application.mappers.evaluation_mapper import (
    EvaluationMapper,
)


class DeleteEvaluationUseCase:
    """
    Delete an evaluation result.
    """

    def __init__(
        self,
        evaluation_repo: EvaluationRepository,
    ) -> None:

        self._evaluation_repo = evaluation_repo

    def execute(
        self,
        request: DeleteEvaluationRequest,
    ) -> EvaluationDeletedResponse:

        evaluation = self._evaluation_repo.get_by_id(
            request.evaluation_id
        )

        self._evaluation_repo.delete(request.evaluation_id)

        return EvaluationMapper.to_deleted(
            evaluation
        )