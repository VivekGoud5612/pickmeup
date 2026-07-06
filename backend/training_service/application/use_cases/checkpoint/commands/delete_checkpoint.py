from __future__ import annotations

from training_service.application.dto.checkpoint.requests import (
    DeleteCheckpointRequest,
)
from training_service.application.dto.checkpoint.responses import (
    DeleteCheckpointResponse,
)
from training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)


class DeleteCheckpointUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo

    def execute(
        self,
        request: DeleteCheckpointRequest,
    ) -> DeleteCheckpointResponse:

        checkpoint = self._checkpoint_repo.get_by_id(
            request.checkpoint_id
        )

        self._checkpoint_repo.delete(checkpoint)

        return CheckpointMapper.to_deleted(checkpoint)