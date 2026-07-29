from __future__ import annotations

from training_service.application.dto.training.requests import (
    DeleteTrainingRequest,
)
from training_service.application.dto.training.responses import (
    TrainingDeletedResponse,
)
from training_service.application.repositories.training_repository import (
    TrainingRepository,
)
from training_service.application.mappers.training_mapper import (
    TrainingMapper,
)
from engine_gateway.infrastructure.engine_registry import (
    EngineRegistry,
)


class DeleteTrainingUseCase:
    """
    Delete a training run.
    """

    def __init__(
        self,
        training_repo: TrainingRepository,
        engine_registry : EngineRegistry,
    ) -> None:

        self._training_repo = training_repo
        self._engine_registry

    def execute(
        self,
        request: DeleteTrainingRequest,
    ) -> TrainingDeletedResponse:

        self._engine_registry.remove(request.training_run_id)

        self._training_repo.delete(request.training_run_id)

        return TrainingMapper.to_deleted(
            training
        )