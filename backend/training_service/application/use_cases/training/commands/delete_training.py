from __future__ import annotations

from backend.engine_gateway.infrastructure.engine_registry import EngineRegistry
from backend.training_service.application.dto.training.requests import DeleteTrainingRequest
from backend.training_service.application.dto.training.responses import TrainingDeletedResponse
from backend.training_service.application.mappers.training_mapper import TrainingMapper
from backend.training_service.application.repositories.training_repository import TrainingRunRepository


class DeleteTrainingUseCase:
    def __init__(self, training_repo: TrainingRunRepository, engine_registry: EngineRegistry) -> None:
        self._training_repo = training_repo
        self._engine_registry = engine_registry

    def execute(self, request: DeleteTrainingRequest) -> TrainingDeletedResponse:
        training_run = self._training_repo.get_by_id(request.training_run_id)
        self._engine_registry.remove(request.training_run_id)
        self._training_repo.delete(request.training_run_id)
        return TrainingMapper.to_deleted(training_run)
