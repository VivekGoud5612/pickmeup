from __future__ import annotations

from backend.training_service.application.dto.checkpoint.requests import (
    GetBestCheckpointRequest,
)
from backend.training_service.application.dto.checkpoint.responses import (
    CheckpointSummaryResponse,
)
from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from backend.training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)


class GetBestCheckpointUseCase:

    def __init__(
        self,
        training_repo: TrainingRunRepository,
        checkpoint_repo: CheckpointRepository,
    ) -> None:

        self._training_repo = training_repo
        self._checkpoint_repo = checkpoint_repo

    def execute(
        self,
        request: GetBestCheckpointRequest,
    ) -> CheckpointSummaryResponse:

        training = self._training_repo.get_by_id(
            request.training_run_id,
        )

        if training.best_checkpoint_id is None:
            raise ValueError(
                "No best checkpoint exists for this training run."
            )

        checkpoint = self._checkpoint_repo.get_by_id(
            training.best_checkpoint_id,
        )

        return CheckpointMapper.to_summary(
            checkpoint,
        )