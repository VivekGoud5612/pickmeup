from __future__ import annotations
from backend.training_service.domain.entities.checkpoint import (
    TrainingCheckpoint,
)

from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from backend.training_service.application.repositories.training_configuration_repository import (
    TrainingConfigurationRepository,
)

from backend.training_service.application.dto.checkpoint.requests import (
    SaveCheckpointRequest,
)

from backend.training_service.application.dto.checkpoint.responses import (
    CheckpointCreatedResponse,
)
from backend.training_service.application.mappers.checkpoint_mapper import (
    CheckpointMapper,
)
from backend.engine_gateway.infrastructure.engine_registry import (
    EngineRegistry,
)
from backend.contracts.engine.dto.requests import (
    SaveEngineCheckpointRequest,
)
from backend.contracts.engine.dto.responses import (
    EngineCheckpointSavedResponse,
)



class SaveCheckpointUseCase:

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
        training_repo: TrainingRunRepository,
        training_config_repo: TrainingConfigurationRepository,
        engine_registry: EngineRegistry,
    ):

        self._checkpoint_repo = checkpoint_repo
        self._training_repo = training_repo
        self._training_config_repo = training_config_repo
        self._engine_registry = engine_registry

    def execute(
        self,
        request : SaveCheckpointRequest,
        ) -> CheckpointCreatedResponse:

        training_run = self._training_repo.get_by_id(request.training_run_id)
        configuration = self._training_config_repo.get_by_id(training_run.configuration_id)

        engine_client = self._engine_registry.get(request.training_run_id)
        engine_request = SaveEngineCheckpointRequest()

        engine_response = engine_client.save_checkpoint(engine_request)

        checkpoint = self._create_checkpoint(
            request, training_run, configuration, engine_response
        )

        self._checkpoint_repo.save(checkpoint)

        training_run.attach_checkpoint(checkpoint.id)

        self._training_repo.update(training_run)

        return CheckpointMapper.to_created(checkpoint)


    def _create_checkpoint(
        self,
        request: SaveCheckpointRequest,
        training_run : TrainingRun,
        configuration,
        engine_response: EngineCheckpointSavedResponse,
    ) -> TrainingCheckpoint:

        return TrainingCheckpoint(
            training_run_id=training_run.id,
            progress=training_run.progress,
            checkpoint_type=engine_response.checkpoint_type,
            file_path=engine_response.checkpoint_path,
            hyperparameters=configuration.hyperparameters,
            reward_weights=configuration.reward_weights,
            curriculum_settings=configuration.curriculum,
            created_at=engine_response.created_at,
        )
