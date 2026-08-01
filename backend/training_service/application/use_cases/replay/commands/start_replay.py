from __future__ import annotations

from backend.engine_gateway.infrastructure.engine_registry import (
    EngineRegistry,
)

from backend.training_service.application.dto.replay.requests import (
    ReplayCheckpointRequest,
)

from backend.training_service.application.dto.replay.responses import (
    ReplayStartedResponse,
)

from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)


class StartReplayUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
        engine_registry: EngineRegistry,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo
        self._engine_registry = engine_registry

    def execute(
        self,
        request: ReplayCheckpointRequest,
    ) -> ReplayStartedResponse:

        checkpoint = self._checkpoint_repo.get_by_id(
            request.checkpoint_id,
        )

        engine_client = self._engine_registry.get(
            checkpoint.training_run_id,
        )

        engine_client.replay(
            checkpoint_path=checkpoint.file_path,
            episodes=request.replay_episodes,
        )

        return ReplayStartedResponse()