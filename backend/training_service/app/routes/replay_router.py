from __future__ import annotations

from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
)
from sqlalchemy.orm import Session

from backend.training_service.infrastructure.database.dependencies import (
    get_session,
)

from backend.training_service.infrastructure.repositories.sql_checkpoint_repository import (
    SQLCheckpointRepository,
)

from backend.training_service.application.dto.replay.requests import (
    ReplayCheckpointRequest,
)

from backend.training_service.application.dto.replay.responses import (
    ReplayStartedResponse,
)

from backend.training_service.application.use_cases.replay.commands.start_replay import (
    StartReplayUseCase,
)

from backend.engine_gateway.infrastructure.clients.local_replay_engine_client import (
    ReplayEngineClient,
)


router = APIRouter(
    prefix="/replay",
    tags=["Replay"],
)


@router.post(
    "/{checkpoint_id}",
    response_model=ReplayStartedResponse,
)
def start_replay(
    checkpoint_id: UUID,
    body: ReplayCheckpointRequest,
    session: Session = Depends(get_session),
) -> ReplayStartedResponse:

    checkpoint_repo = SQLCheckpointRepository(session)

    replay_client = ReplayEngineClient()

    usecase = StartReplayUseCase(
        checkpoint_repo=checkpoint_repo,
        replay_client=replay_client,
    )

    return usecase.execute(
        ReplayCheckpointRequest(
            checkpoint_id=checkpoint_id,
            replay_episodes=body.replay_episodes,
        )
    )