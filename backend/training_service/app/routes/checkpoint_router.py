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
from backend.training_service.infrastructure.repositories.sql_training_configuration_repository import (
    SQLTrainingConfigurationRepository,
)
from backend.training_service.infrastructure.repositories.sql_evaluation_repository import (
    SQLEvaluationRepository,
)

from backend.engine_gateway.infrastructure.registry_instance import (
    engine_registry,
)

from backend.training_service.application.dto.checkpoint.requests import (
    SaveCheckpointRequest,
    DeleteCheckpointRequest,
    GetCheckpointRequest,
    GetLatestCheckpointRequest,
    GetBestCheckpointRequest,
    ListCheckpointsRequest,
)

from backend.training_service.application.dto.checkpoint.responses import (
    CheckpointCreatedResponse,
    CheckpointDeletedResponse,
    CheckpointSummaryResponse,
    ListCheckpointsResponse,
)

from backend.training_service.application.use_cases.checkpoint.commands.save_checkpoint import (
    SaveCheckpointUseCase,
)
from backend.training_service.application.use_cases.checkpoint.commands.delete_checkpoint import (
    DeleteCheckpointUseCase,
)

from backend.training_service.application.use_cases.checkpoint.queries.get_checkpoint import (
    GetCheckpointUseCase,
)
from backend.training_service.application.use_cases.checkpoint.queries.get_latest_checkpoint import (
    GetLatestCheckpointUseCase,
)
from backend.training_service.application.use_cases.checkpoint.queries.get_best_checkpoint import (
    GetBestCheckpointUseCase,
)
from backend.training_service.application.use_cases.checkpoint.queries.list_checkpoints import (
    ListCheckpointsUseCase,
)

router = APIRouter(
    prefix="/checkpoints",
    tags=["Checkpoint"],
)



@router.post(
    "/training/{training_run_id}",
    response_model=CheckpointCreatedResponse,
)
def save_checkpoint(
    training_run_id: UUID,
    body: SaveCheckpointRequest,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)
    training_repo = SQLTrainingRunRepository(session)
    configuration_repo = SQLTrainingConfigurationRepository(session)

    usecase = SaveCheckpointUseCase(
        checkpoint_repo=checkpoint_repo,
        training_repo=training_repo,
        training_config_repo=configuration_repo,
        engine_registry=engine_registry,
    )

    return usecase.execute(
        SaveCheckpointRequest(
            training_run_id=training_run_id,
            checkpoint_name=body.checkpoint_name,
            notes=body.notes,
        )
    )


@router.delete(
    "/{checkpoint_id}",
    response_model=CheckpointDeletedResponse,
)
def delete_checkpoint(
    checkpoint_id: UUID,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)
    training_repo = SQLTrainingRunRepository(session)
    evaluation_repo = SQLEvaluationRepository(session)

    usecase = DeleteCheckpointUseCase(
        checkpoint_repo=checkpoint_repo,
        training_repo=training_repo,
        evaluation_repo=evaluation_repo,
        engine_registry=engine_registry,
    )

    return usecase.execute(
        DeleteCheckpointRequest(
            checkpoint_id=checkpoint_id,
        )
    )


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

    return usecase.execute(
        GetCheckpointRequest(
            checkpoint_id=checkpoint_id,
        )
    )


@router.get(
    "/training/{training_run_id}",
    response_model=ListCheckpointsResponse,
)
def list_checkpoints(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)

    usecase = ListCheckpointsUseCase(
        checkpoint_repo=checkpoint_repo,
    )

    return usecase.execute(
        ListCheckpointsRequest(
            training_run_id=training_run_id,
        )
    )


@router.get(
    "/training/{training_run_id}/latest",
    response_model=CheckpointSummaryResponse,
)
def get_latest_checkpoint(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)
    training_repo = SQLTrainingRunRepository(session)

    usecase = GetLatestCheckpointUseCase(
        training_repo=training_repo,
        checkpoint_repo=checkpoint_repo,
    )

    return usecase.execute(
        GetLatestCheckpointRequest(
            training_run_id=training_run_id,
        )
    )


@router.get(
    "/training/{training_run_id}/best",
    response_model=CheckpointSummaryResponse,
)
def get_best_checkpoint(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    checkpoint_repo = SQLCheckpointRepository(session)
    training_repo = SQLTrainingRunRepository(session)

    usecase = GetBestCheckpointUseCase(
        training_repo=training_repo,
        checkpoint_repo=checkpoint_repo,
    )

    return usecase.execute(
        GetBestCheckpointRequest(
            training_run_id=training_run_id,
        )
    )