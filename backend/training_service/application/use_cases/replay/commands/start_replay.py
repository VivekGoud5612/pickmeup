from __future__ import annotations

from backend.training_service.application.dto.replay.requests import (
    ReplayCheckpointRequest,
)

from backend.training_service.application.dto.replay.responses import (
    ReplayStartedResponse,
)

from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)

from backend.engine_gateway.infrastructure.clients.local_replay_engine_client import (
    ReplayEngineClient,
)

from backend.engine_gateway.application.dto.requests import (
    ReplayCheckpointRequest,
)


class StartReplayUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
        replay_client: ReplayEngineClient,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo
        self._replay_client = replay_client

    def execute(
        self,
        request: ReplayCheckpointRequest,
    ) -> ReplayStartedResponse:

        checkpoint = self._checkpoint_repo.get_by_id(
            request.checkpoint_id,
        )

        self._replay_client.replay(
            ReplayCheckpointRequest(
                checkpoint_path=checkpoint.file_path,
                episodes=request.replay_episodes,
            )
        )

        return ReplayStartedResponse()