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

        checkpoint = self._create_checkpoint(request, training_run)

        self._checkpoint_repo.save(checkpoint)

        training_run.attach_checkpoint(checkpoint.id)

        self._training_repo.update(training_run)

        return self._create_response(checkpoint)


    def _create_checkpoint(
        self,
        request: SaveCheckpointRequest,
        training_run,
    ) -> TrainingCheckpoint:

        return TrainingCheckpoint(
            training_run_id=training_run.id,
            configuration_id=training_run.configuration_id,
            progress=training_run.progress,
            checkpoint_type=request.checkpoint_type,
            file_path=request.file_path,
            notes=request.notes,
        )

    def _create_response(
        self,
        checkpoint: TrainingCheckpoint,
    ) -> CheckpointCreatedResponse:

        checkpoint_summary = CheckpointSummaryResponse(
            id = checkpoint.id,
            training_run_id = checkpoint.training_run_id,
            progress = checkpoint.progress,
            created_at = checkpoint.created_at,
        )
        
        return CheckpointMapper.to_created(checkpoint)