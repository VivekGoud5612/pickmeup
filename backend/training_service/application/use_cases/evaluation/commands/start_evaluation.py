from __future__ import annotations

from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from backend.training_service.application.repositories.evaluation_repository import (
    EvaluationRepository,
)

from backend.training_service.application.dto.evaluation.requests import (
    StartEvaluationRequest,
)
from backend.training_service.application.dto.evaluation.responses import (
    EvaluationCreatedResponse,
)

from backend.training_service.domain.entities.checkpoint import (
    TrainingCheckpoint,
)
from backend.training_service.domain.entities.evaluation_result import (
    EvaluationResult,
)

from backend.training_service.application.mappers.evaluation_mapper import (
    EvaluationMapper,
)


class StartEvaluationUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
        evaluation_repo: EvaluationRepository,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo
        self._evaluation_repo = evaluation_repo

    def execute(
        self,
        request: StartEvaluationRequest,
    ) -> EvaluationCreatedResponse:

        checkpoint = self._checkpoint_repo.get_by_id(
            request.checkpoint_id,
        )

        evaluation = self._create_evaluation(
            checkpoint,
            request,
        )

        evaluation.start()

        self._evaluation_repo.save(
            evaluation,
        )

        return EvaluationMapper.to_created(
            evaluation,
        )

    def _create_evaluation(
        self,
        checkpoint: TrainingCheckpoint,
        request: StartEvaluationRequest,
    ) -> EvaluationResult:

        return EvaluationResult(
            checkpoint_id=checkpoint.id,
            num_episodes=request.evaluation_episodes,
            notes=request.notes,
        )