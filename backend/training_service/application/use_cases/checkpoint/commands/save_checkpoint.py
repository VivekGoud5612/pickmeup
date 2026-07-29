from __future__ import annotations
from dataclasses import dataclass 

from training_service.domain.entities.checkpoint import (
    TrainingCheckpoint,
)

from training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)

from training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)

from training_service.application.dto.checkpoint.requests import (
    SaveCheckpointRequest,
)

from training_service.application.dto.checkpoint.responses import (
    CheckpointCreatedResponse, CheckpointSummaryResponse,
)
from training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)
from engine_gateway.infrastructure.engine_registry import (
    EngineRegistry,
)
from engine_gateway.application.dto.requests import (
    EngineCheckpointRequest,
)
from engine_gateway.application.dto.responses import (
    EngineCheckpointResponse,
)



class SaveCheckpointUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
        training_repo: TrainingRepository,

    ):

        self._checkpoint_repo = checkpoint_repo
        self._training_repo = training_repo

    def execute(
        self,
        request : SaveCheckpointRequest,
        ) -> CheckpointCreatedResponse:

        training_run = self._training_repo.get_by_id(request.training_run_id)

        engine_client = self._engine_registry.get(request.training_run_id)
        engine_request = EngineCheckpointRequest(
            checkpoint_name = request.checkpoint_name,
        )

        engine_response = engine_client.save_checkpoint(engine_request)

        checkpoint = self._create_checkpoint(request, training_run, engine_response)

        self._checkpoint_repo.save(checkpoint)

        training_run.attach_checkpoint(checkpoint.id)

        self._training_repo.update(training_run)

        return CheckpointMapper.to_created(checkpoint)


    def _create_checkpoint(
        self,
        request: SaveCheckpointRequest,
        training_run : TrainingRun,
        engine_response : EngineCheckpointResponse,
    ) -> TrainingCheckpoint:

        return TrainingCheckpoint(
            training_run_id=training_run.id,
            configuration_id=training_run.configuration_id,
            progress=training_run.progress,
            checkpoint_type = engine_response.checkpoint_type,
            file_path=engine_response.checkpoint_path,
            notes=request.notes,
            created_at = engine_response.created_at,
        )
