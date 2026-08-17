from __future__ import annotations

from backend.contracts.engine.dto.requests import StopEngineTrainingRequest
from backend.engine_gateway.infrastructure.engine_registry import EngineRegistry
from backend.training_service.application.dto.training.requests import StopTrainingRequest
from backend.training_service.application.dto.training.responses import TrainingSummaryResponse
from backend.training_service.application.mappers.training_mapper import TrainingMapper
from backend.training_service.application.repositories.training_repository import TrainingRunRepository


class StopTrainingUseCase:
    def __init__(self, training_repo: TrainingRunRepository, engine_registry: EngineRegistry) -> None:
        self._training_repo = training_repo
        self._engine_registry = engine_registry

    def execute(self, request: StopTrainingRequest) -> TrainingSummaryResponse:
        training_run = self._training_repo.get_by_id(request.training_run_id)
        engine_client = self._engine_registry.get(request.training_run_id)
        engine_client.stop(StopEngineTrainingRequest())
        training_run.stop()
        self._training_repo.update(training_run)
        self._engine_registry.remove(request.training_run_id)
        return TrainingMapper.to_summary(training_run)
