from __future__ import annotations

from pathlib import Path

from backend.training_service.domain.entities.checkpoint import (
    TrainingCheckpoint,
)

from backend.training_service.domain.value_objects import (
    TrainingProgress,
)

from backend.training_service.infrastructure.models.training_checkpoint_model import (
    TrainingCheckpointModel,
)


class TrainingCheckpointPersistenceMapper:
    """
    Maps between TrainingCheckpoint domain entities
    and SQLAlchemy models.
    """

    @staticmethod
    def to_model(
        checkpoint: TrainingCheckpoint,
    ) -> TrainingCheckpointModel:

        return TrainingCheckpointModel(

            # -------------------------
            # Identity
            # -------------------------

            id=checkpoint.id,

            training_run_id=checkpoint.training_run_id,

            configuration_id=checkpoint.configuration_id,

            # -------------------------
            # Progress
            # -------------------------

            episode=checkpoint.progress.episode,

            step=checkpoint.progress.step,

            # -------------------------
            # Metadata
            # -------------------------

            checkpoint_type=checkpoint.checkpoint_type,

            file_path=str(checkpoint.file_path),

            is_best=checkpoint.is_best,

            description=checkpoint.description,

            created_at=checkpoint.created_at,
        )

    @staticmethod
    def to_entity(
        model: TrainingCheckpointModel,
    ) -> TrainingCheckpoint:

        return TrainingCheckpoint(

            # -------------------------
            # Identity
            # -------------------------

            id=model.id,

            training_run_id=model.training_run_id,

            configuration_id=model.configuration_id,

            # -------------------------
            # Progress
            # -------------------------

            progress=TrainingProgress(
                episode=model.episode,
                step=model.step,
            ),

            # -------------------------
            # Metadata
            # -------------------------

            checkpoint_type=model.checkpoint_type,

            file_path=Path(model.file_path),

            is_best=model.is_best,

            description=model.description,

            created_at=model.created_at,
        )