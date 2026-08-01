from __future__ import annotations

from backend.training_service.application.event_handlers.base import (
    EngineEventHandler,
)

from backend.engine_gateway.application.events.checkpoint_events import (
    CheckpointCreatedEvent,
)

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)

from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)

from backend.training_service.domain.entities.checkpoint import (
    TrainingCheckpoint,
)

from backend.training_service.domain.value_objects import (
    TrainingProgress,
)


class CheckpointCreatedHandler(EngineEventHandler):

    def __init__(
        self,
        training_repo: TrainingRunRepository,
        checkpoint_repo: CheckpointRepository,
    ) -> None:

        self._training_repo = training_repo
        self._checkpoint_repo = checkpoint_repo

    def handle(
        self,
        event: CheckpointCreatedEvent,
    ) -> None:

        training = self._training_repo.get_by_id(
            event.run_id,
        )

        checkpoint = TrainingCheckpoint(

            training_run_id=training.id,

            progress=TrainingProgress(
                step=event.global_step,
            ),

            hyperparameters=event.hyperparameters,

            reward_weights=event.reward_weights,

            curriculum_settings=event.curriculum_settings,

            checkpoint_type=event.checkpoint_type,

            file_path=event.checkpoint_path,

            created_at=event.created_at,
        )

        self._checkpoint_repo.save(
            checkpoint,
        )

        training.attach_latest_checkpoint(
            checkpoint.id,
        )

        self._training_repo.update(
            training,
        )