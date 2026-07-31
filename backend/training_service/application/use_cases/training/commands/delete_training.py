from __future__ import annotations

from backend.training_service.application.dto.training.requests import (
    DeleteTrainingRequest,
)
from backend.training_service.application.dto.training.responses import (
    TrainingDeletedResponse,
)
from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from backend.training_service.application.mappers.training_mapper import (
    TrainingMapper,
)
from backend.engine_gateway.infrastructure.dummyengine_registry import (
    DummyEngineRegistry,
)


class DeleteTrainingUseCase:
    """
    Delete a training run.
    """

    def __init__(
        self,
        training_repo: TrainingRunRepository,
        engine_registry: DummyEngineRegistry,
    ) -> None:

        self._training_repo = training_repo
        self._engine_registry = engine_registry

    def execute(
        self,
        request: DeleteTrainingRequest,
    ) -> TrainingDeletedResponse:

        training = self._training_repo.get_by_id(
            request.training_run_id,
        )

        if self._engine_registry.exists(
            request.training_run_id,
        ):
            self._engine_registry.remove(
                request.training_run_id,
            )

        self._training_repo.delete(
            request.training_run_id,
        )

        return TrainingMapper.to_deleted(
            training,
        )