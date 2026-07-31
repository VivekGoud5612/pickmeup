from __future__ import annotations

from backend.training_service.application.dto.training.requests import (
    PauseTrainingRequest,
)
from backend.training_service.application.dto.training.responses import (
    TrainingSummaryResponse,
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


class PauseTrainingUseCase:
    """
    Pause a running training session.
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
        request: PauseTrainingRequest,
    ) -> TrainingSummaryResponse:

        training_run = self._training_repo.get_by_id(
            request.training_run_id,
        )

        engine_client = self._engine_registry.get(
            request.training_run_id,
        )

        engine_client.pause()

        training_run.pause()

        self._training_repo.update(
            training_run,
        )

        return TrainingMapper.to_summary(
            training_run,
        )