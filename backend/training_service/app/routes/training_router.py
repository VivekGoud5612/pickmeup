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

from backend.training_service.infrastructure.repositories.sql_training_run_repository import (
    SQLTrainingRunRepository,
)
from backend.training_service.infrastructure.repositories.sql_training_configuration_repository import (
    SQLTrainingConfigurationRepository,
)
from backend.training_service.infrastructure.repositories.sql_checkpoint_repository import (
    SQLCheckpointRepository,
)
from backend.training_service.infrastructure.repositories.sql_evaluation_repository import (
    SQLEvaluationRepository,
)

from backend.engine_gateway.infrastructure.registry_instance import (
    engine_registry,
)

from backend.training_service.application.dto.training.requests import (
    StartTrainingRequest,
    PauseTrainingRequest,
    ResumeTrainingRequest,
    StopTrainingRequest,
    DeleteTrainingRequest,
    GetTrainingSummaryRequest,
    ListTrainingRunsRequest,
)

from backend.training_service.application.dto.training.responses import (
    TrainingCreatedResponse,
    TrainingSummaryResponse,
    TrainingDeletedResponse,
    ListTrainingRunsResponse,
)

from backend.training_service.application.use_cases.training.commands.start_training import (
    StartTrainingUseCase,
)
from backend.training_service.application.use_cases.training.commands.pause_training import (
    PauseTrainingUseCase,
)
from backend.training_service.application.use_cases.training.commands.resume_training import (
    ResumeTrainingUseCase,
)
from backend.training_service.application.use_cases.training.commands.stop_training import (
    StopTrainingUseCase,
)
from backend.training_service.application.use_cases.training.commands.delete_training import (
    DeleteTrainingUseCase,
)

from backend.training_service.application.use_cases.training.queries.get_training_summary import (
    GetTrainingSummaryUseCase,
)
from backend.training_service.application.use_cases.training.queries.list_training_runs import (
    ListTrainingRunsUseCase,
)

router = APIRouter(
    prefix="/training",
    tags=["Training"],
)


@router.post(
    "/start",
    response_model=TrainingCreatedResponse,
)
def start_training(
    request: StartTrainingRequest,
    session: Session = Depends(get_session),
):

    training_repo = SQLTrainingRunRepository(session)
    configuration_repo = SQLTrainingConfigurationRepository(session)
    checkpoint_repo = SQLCheckpointRepository(session)
    evaluation_repo = SQLEvaluationRepository(session)

    usecase = StartTrainingUseCase(
        training_repo=training_repo,
        training_config_repo=configuration_repo,
        checkpoint_repo=checkpoint_repo,
        evaluation_repo=evaluation_repo,
        engine_registry=engine_registry,
    )

    return usecase.execute(request)


@router.post(
    "/{training_run_id}/pause",
    response_model=TrainingSummaryResponse,
)
def pause_training(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    training_repo = SQLTrainingRunRepository(session)

    usecase = PauseTrainingUseCase(
        training_repo=training_repo,
        engine_registry=engine_registry,
    )

    return usecase.execute(
        PauseTrainingRequest(
            training_run_id=training_run_id,
        )
    )


@router.post(
    "/{training_run_id}/resume",
    response_model=TrainingSummaryResponse,
)
def resume_training(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    training_repo = SQLTrainingRunRepository(session)

    usecase = ResumeTrainingUseCase(
        training_repo=training_repo,
        engine_registry=engine_registry,
    )

    return usecase.execute(
        ResumeTrainingRequest(
            training_run_id=training_run_id,
        )
    )


@router.post(
    "/{training_run_id}/stop",
    response_model=TrainingSummaryResponse,
)
def stop_training(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    training_repo = SQLTrainingRunRepository(session)

    usecase = StopTrainingUseCase(
        training_repo=training_repo,
        engine_registry=engine_registry,
    )

    return usecase.execute(
        StopTrainingRequest(
            training_run_id=training_run_id,
        )
    )


@router.delete(
    "/{training_run_id}",
    response_model=TrainingDeletedResponse,
)
def delete_training(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    training_repo = SQLTrainingRunRepository(session)
    config_repo = SQLTrainingConfigurationRepository(session)
    checkpoint_repo = SQLCheckpointRepository(session)
    evaluation_repo = SQLEvaluationRepository(session)

    usecase = DeleteTrainingUseCase(
        training_repo=training_repo,
        config_repo=config_repo,
        checkpoint_repo=checkpoint_repo,
        evaluation_repo=evaluation_repo,
        engine_registry=engine_registry,
    )

    return usecase.execute(
        DeleteTrainingRequest(
            training_run_id=training_run_id,
        )
    )


@router.get(
    "/{training_run_id}",
    response_model=TrainingSummaryResponse,
)
def get_training(
    training_run_id: UUID,
    session: Session = Depends(get_session),
):

    training_repo = SQLTrainingRunRepository(session)

    usecase = GetTrainingSummaryUseCase(
        training_repo=training_repo,
    )

    return usecase.execute(
        GetTrainingSummaryRequest(
            training_run_id=training_run_id,
        )
    )


@router.get(
    "",
    response_model=ListTrainingRunsResponse,
)
def list_training_runs(
    session: Session = Depends(get_session),
):

    training_repo = SQLTrainingRunRepository(session)

    usecase = ListTrainingRunsUseCase(
        training_repo=training_repo,
    )

    return usecase.execute(
        ListTrainingRunsRequest()
    )