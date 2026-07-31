from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.training_service.infrastructure.database.dependencies import (
    get_session,
)

from backend.training_service.infrastructure.repositories.sql_checkpoint_repository import (
    SQLCheckpointRepository,
)
from backend.training_service.infrastructure.repositories.sql_training_run_repository import (
    SQLTrainingRunRepository,
)
from backend.training_service.infrastructure.repositories.sql_evaluation_repository import (
    SQLEvaluationRepository,
)

from backend.engine_gateway.infrastructure.dummyengine_registry import (
    DummyEngineRegistry,
)

from backend.training_service.application.dto.checkpoint.requests import (
    SaveCheckpointRequest,
    DeleteCheckpointRequest,
    GetBestCheckpointRequest,
    GetCheckpointRequest,
    ListCheckpointsRequest,
)

from backend.training_service.application.dto.checkpoint.responses import (
    CheckpointCreatedResponse,
    CheckpointDeletedResponse,
    CheckpointSummaryResponse,
)

from backend.training_service.application.dto.evaluation.requests import (
    StartEvaluationRequest,
)

from backend.training_service.application.dto.evaluation.responses import (
    EvaluationCreatedResponse,
)

from backend.training_service.application.use_cases.checkpoint.commands.save_checkpoint import (
    SaveCheckpointUseCase,
)
from backend.training_service.application.use_cases.checkpoint.commands.delete_checkpoint import (
    DeleteCheckpointUseCase,
)

from backend.training_service.application.use_cases.checkpoint.queries.get_best_checkpoint import (
    GetBestCheckpointUseCase,
)
from backend.training_service.application.use_cases.checkpoint.queries.get_checkpoint import (
    GetCheckpointUseCase,
)
from backend.training_service.application.use_cases.checkpoint.queries.list_checkpoints import (
    ListCheckpointsUseCase,
)

from backend.training_service.application.use_cases.evaluation.commands.start_evaluation import (
    StartEvaluationUseCase,
)

router = APIRouter(
    prefix="/checkpoints",
    tags=["Checkpoint"],
)

engine_registry = DummyEngineRegistry()


@router.post(
    "/training/{training_run_id}",
    response_model=CheckpointCreatedResponse,
)
def save_checkpoint(
    training_run_id: UUID,
    request: SaveCheckpointRequest,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)
    training_repo = SQLTrainingRunRepository(session)

    usecase = SaveCheckpointUseCase(
        checkpoint_repo=checkpoint_repo,
        training_repo=training_repo,
        engine_registry=engine_registry,
    )

    request.training_run_id = training_run_id

    return usecase.execute(request)


@router.delete(
    "/{checkpoint_id}",
    response_model=CheckpointDeletedResponse,
)
def delete_checkpoint(
    checkpoint_id: UUID,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)

    usecase = DeleteCheckpointUseCase(
        checkpoint_repo=checkpoint_repo,
    )

    request = DeleteCheckpointRequest(
        checkpoint_id=checkpoint_id,
    )

    return usecase.execute(request)


@router.get(
    "/{checkpoint_id}",
    response_model=CheckpointSummaryResponse,
)
def get_checkpoint(
    checkpoint_id: UUID,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)

    usecase = GetCheckpointUseCase(
        checkpoint_repo=checkpoint_repo,
    )

    request = GetCheckpointRequest(
        checkpoint_id=checkpoint_id,
    )

    return usecase.execute(request)


@router.get(
    "/training/{training_run_id}",
    response_model=list[CheckpointSummaryResponse],
)
def list_checkpoints(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)

    usecase = ListCheckpointsUseCase(
        checkpoint_repo=checkpoint_repo,
    )

    request = ListCheckpointsRequest(
        training_run_id=training_run_id,
    )

    return usecase.execute(request)


@router.get(
    "/training/{training_run_id}/best",
    response_model=CheckpointSummaryResponse,
)
def get_best_checkpoint(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)

    usecase = GetBestCheckpointUseCase(
        checkpoint_repo=checkpoint_repo,
    )

    request = GetBestCheckpointRequest(
        training_run_id=training_run_id,
    )

    return usecase.execute(request)


@router.post(
    "/{checkpoint_id}/evaluate",
    response_model=EvaluationCreatedResponse,
)
def evaluate_checkpoint(
    checkpoint_id: UUID,
    request: StartEvaluationRequest,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)
    evaluation_repo = SQLEvaluationRepository(session)

    usecase = StartEvaluationUseCase(
        checkpoint_repo=checkpoint_repo,
        evaluation_repo=evaluation_repo,
    )

    request.checkpoint_id = checkpoint_id

    return usecase.execute(request)