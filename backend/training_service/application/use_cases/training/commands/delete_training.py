from __future__ import annotations

from training_service.application.dto.training.requests import (
    DeleteTrainingRequest,
)
from training_service.application.dto.training.responses import (
    DeleteTrainingResponse,
)
from training_service.application.repositories.training_repository import (
    TrainingRepository,
)
from training_service.application.mappers.training_mapper import (
    TrainingMapper,
)


class DeleteTrainingUseCase:
    """
    Delete a training run.
    """

    def __init__(
        self,
        training_repo: TrainingRepository,
    ) -> None:

        self._training_repo = training_repo

    def execute(
        self,
        request: DeleteTrainingRequest,
    ) -> DeleteTrainingResponse:

        training = self._training_repo.get_by_id(
            request.training_run_id
        )

        self._training_repo.delete(training)

        return TrainingMapper.to_deleted(
            training
        )