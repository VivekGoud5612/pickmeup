from __future__ import annotations

from backend.training_service.application.dto.checkpoint.requests import (
    DeleteCheckpointRequest,
)
from backend.training_service.application.dto.checkpoint.responses import (
    CheckpointDeletedResponse,
)
from backend.training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)
from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from backend.training_service.application.repositories.evaluation_repository import (
    EvaluationRepository,
)

from backend.engine_gateway.infrastructure.engine_registry import (
    EngineRegistry,
)

from backend.engine_gateway.application.dto.requests import (
    DeleteEngineCheckpointRequest,
)


class DeleteCheckpointUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
        training_repo: TrainingRunRepository,
        evaluation_repo: EvaluationRepository,
        engine_registry: EngineRegistry,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo
        self._training_repo = training_repo
        self._evaluation_repo = evaluation_repo
        self._engine_registry = engine_registry

    def execute(
        self,
        request: DeleteCheckpointRequest,
    ) -> CheckpointDeletedResponse:

        checkpoint = self._checkpoint_repo.get_by_id(
            request.checkpoint_id,
        )

        training = self._training_repo.get_by_id(
            checkpoint.training_run_id,
        )

        if training.latest_checkpoint_id == checkpoint.id:

            training.detach_latest_checkpoint()

            self._training_repo.update(
                training,
            )

        if self._engine_registry.exists(
            training.id,
        ):

            client = self._engine_registry.get(
                training.id,
            )

            client.delete_checkpoint(
                DeleteEngineCheckpointRequest(
                    checkpoint_path=checkpoint.file_path,
                )
            )

        # Delete all evaluations associated with this checkpoint
        evaluations = self._evaluation_repo.list_by_checkpoint(
            checkpoint.id,
        )

        for evaluation in evaluations:
            self._evaluation_repo.delete(
                evaluation.id,
            )

        self._checkpoint_repo.delete(
            checkpoint.id,
        )

        return CheckpointMapper.to_deleted(
            checkpoint,
        )