from __future__ import annotations

from training_service.application.dto.training.requests import (
    StopTrainingRequest,
)
from training_service.application.dto.training.responses import (
    TrainingSummaryResponse,
)
from training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from training_service.application.mappers.training_mapper import (
    TrainingMapper,
)
from engine_gateway.infrastructure.engine_registry import (
    EngineRegistry,
)


class StopTrainingUseCase:

    def __init__(
        self,
        training_repo: TrainingRunRepository,
        engine_registry : EngineRegistry,
    ) -> None:
        self._training_repo = training_repo
        self._engine_registry

    def execute(
        self,
        request: StopTrainingRequest,
    ) -> TrainingSummaryResponse:

        training_run = self._training_repo.get_by_id(
            request.training_run_id
        )

        engine_client = self._engine_registry.get(request.training_run_id)
        engine_client.stop()

        training_run.stop()

        self._training_repo.update(training_run)

        self._engine_registry.remove(request.training_run_id)   ### Remove the stopped training run from the runtime registry of running clients.

        return TrainingMapper.to_summary(training_run)

    
