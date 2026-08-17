from __future__ import annotations

from backend.contracts.engine.dto.requests import PauseEngineTrainingRequest
from backend.engine_gateway.infrastructure.engine_registry import EngineRegistry
from backend.training_service.application.dto.training.requests import PauseTrainingRequest
from backend.training_service.application.dto.training.responses import TrainingSummaryResponse
from backend.training_service.application.mappers.training_mapper import TrainingMapper
from backend.training_service.application.repositories.training_repository import TrainingRunRepository


class PauseTrainingUseCase:
    def __init__(self, training_repo: TrainingRunRepository, engine_registry: EngineRegistry) -> None:
        self._training_repo = training_repo
        self._engine_registry = engine_registry

    def execute(self, request: PauseTrainingRequest) -> TrainingSummaryResponse:
        training_run = self._training_repo.get_by_id(request.training_run_id)
        engine_client = self._engine_registry.get(request.training_run_id)
        engine_client.pause(PauseEngineTrainingRequest())
        training_run.pause()
        self._training_repo.update(training_run)
        return TrainingMapper.to_summary(training_run)
