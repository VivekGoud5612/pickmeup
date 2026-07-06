from __future__ import annotations

from training_service.application.dto.checkpoint.requests import (
    GetBestCheckpointRequest,
)
from training_service.application.dto.checkpoint.responses import (
    CheckpointSummaryResponse,
)
from training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)



class GetBestCheckpointUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo

    def execute(
        self,
        request: GetBestCheckpointRequest,
    ) -> CheckpointSummaryResponse:

        checkpoint = self._checkpoint_repo.get_best(
            request.training_run_id
        )

        return CheckpointMapper.to_summary(checkpoint)