from __future__ import annotations

from backend.training_service.application.dto.training.requests import (
    DeleteTrainingRequest,
)
from backend.training_service.application.dto.training.responses import (
    TrainingDeletedResponse,
)

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from backend.training_service.application.repositories.training_configuration_repository import (
    TrainingConfigurationRepository,
)
from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from backend.training_service.application.repositories.evaluation_repository import (
    EvaluationRepository,
)

from backend.training_service.application.mappers.training_mapper import (
    TrainingMapper,
)

from backend.engine_gateway.infrastructure.engine_registry import (
    EngineRegistry,
)

from backend.engine_gateway.application.dto.requests import (
    DeleteEngineCheckpointRequest,
)


class DeleteTrainingUseCase:
    """
    Deletes an entire training run together with every
    checkpoint, evaluation and configuration.
    """

    def __init__(
        self,
        training_repo: TrainingRunRepository,
        config_repo: TrainingConfigurationRepository,
        checkpoint_repo: CheckpointRepository,
        evaluation_repo: EvaluationRepository,
        engine_registry: EngineRegistry,
    ) -> None:

        self._training_repo = training_repo
        self._config_repo = config_repo
        self._checkpoint_repo = checkpoint_repo
        self._evaluation_repo = evaluation_repo
        self._engine_registry = engine_registry

    def execute(
        self,
        request: DeleteTrainingRequest,
    ) -> TrainingDeletedResponse:

        training = self._training_repo.get_by_id(
            request.training_run_id,
        )

        training.detach_latest_checkpoint()

        self._training_repo.update(training)

        checkpoints = self._checkpoint_repo.list_by_run(
            request.training_run_id,
        )

        engine_client = None

        if self._engine_registry.exists(
            request.training_run_id,
        ):
            engine_client = self._engine_registry.get(
                request.training_run_id,
            )

        for checkpoint in checkpoints:

            evaluations = self._evaluation_repo.list_by_checkpoint(
                checkpoint.id,
            )

            for evaluation in evaluations:
                self._evaluation_repo.delete(
                    evaluation.id,
                )

            if engine_client is not None:

                engine_client.delete_checkpoint(
                    DeleteEngineCheckpointRequest(
                        checkpoint_path=checkpoint.file_path,
                    )
                )

            self._checkpoint_repo.delete(
                checkpoint.id,
            )

        self._training_repo.delete(
            training.id,
        )

        self._config_repo.delete(
            training.configuration_id,
        )

        if engine_client is not None:

            self._engine_registry.remove(
                training.id,
            )

        return TrainingMapper.to_deleted(
            training,
        )