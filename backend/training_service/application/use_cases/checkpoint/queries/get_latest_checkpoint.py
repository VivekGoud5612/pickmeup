from __future__ import annotations

from backend.training_service.application.dto.checkpoint.requests import (
    GetLatestCheckpointRequest,
)
from backend.training_service.application.dto.checkpoint.responses import (
    CheckpointSummaryResponse,
)
from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from backend.training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)



class GetLatestCheckpointUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo

    def execute(
        self,
        request: GetLatestCheckpointRequest,
    ) -> CheckpointSummaryResponse:

        checkpoint = self._checkpoint_repo.get_latest(
            request.training_run_id
        )

        return CheckpointMapper.to_summary(checkpoint)
   