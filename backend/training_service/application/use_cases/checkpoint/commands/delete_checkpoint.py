from __future__ import annotations

from backend.training_service.application.dto.checkpoint.requests import (
    DeleteCheckpointRequest,
)
from backend.training_service.application.dto.checkpoint.responses import (
    CheckpointDeletedResponse,
)
from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from backend.training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)
from backend.engine_gateway.infrastructure.engine_registry import (
    EngineRegistry,
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
    ) -> CheckpointDeletedResponse:

        checkpoint = self._checkpoint_repo.get_by_id(
            request.checkpoint_id
        )

        self._checkpoint_repo.delete(checkpoint.id)

        return CheckpointMapper.to_deleted(checkpoint)
