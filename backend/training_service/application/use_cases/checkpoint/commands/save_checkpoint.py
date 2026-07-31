from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from backend.training_service.application.dto.checkpoint.requests import (
    SaveCheckpointRequest,
)
from backend.training_service.application.dto.checkpoint.responses import (
    CheckpointCreatedResponse,
)
from backend.training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)
from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from backend.training_service.domain.entities.checkpoint import (
    TrainingCheckpoint,
)
from backend.training_service.domain.entities.training_run import (
    TrainingRun,
)
from backend.training_service.domain.enums import CheckpointType


class SaveCheckpointUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
        training_repo: TrainingRunRepository,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo
        self._training_repo = training_repo

    def execute(
        self,
        request: SaveCheckpointRequest,
    ) -> CheckpointCreatedResponse:

        training_run = self._training_repo.get_by_id(
            request.training_run_id
        )

        checkpoint = self._create_checkpoint(
            request=request,
            training_run=training_run,
        )

        self._checkpoint_repo.save(
            checkpoint
        )

        training_run.attach_checkpoint(
            checkpoint.id
        )

        self._training_repo.update(
            training_run
        )

        return CheckpointMapper.to_created(
            checkpoint
        )

    def _create_checkpoint(
        self,
        request: SaveCheckpointRequest,
        training_run: TrainingRun,
    ) -> TrainingCheckpoint:

        return TrainingCheckpoint(

            training_run_id=training_run.id,

            configuration_id=training_run.configuration_id,

            progress=training_run.progress,

            checkpoint_type=CheckpointType.MANUAL,

            file_path=Path(
                f"/tmp/{request.checkpoint_name}.pt"
            ),

            description=request.notes,

            created_at=datetime.now(UTC),
        )