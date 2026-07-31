from __future__ import annotations

from backend.training_service.application.dto.checkpoint.requests import (
    ListCheckpointsRequest,
)
from backend.training_service.application.dto.checkpoint.responses import (
    ListCheckpointsResponse,
)
from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from backend.training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)



class ListCheckpointsUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo

    def execute(
        self,
        request: ListCheckpointsRequest,
    ) -> ListCheckpointsResponse:

        checkpoints = self._checkpoint_repo.list_all(
            request.training_run_id
        )

        return ListCheckpointsResponse(
            checkpoints_summary=[
                CheckpointMapper.to_summary(checkpoint)
                for checkpoint in checkpoints
            ]
        )
