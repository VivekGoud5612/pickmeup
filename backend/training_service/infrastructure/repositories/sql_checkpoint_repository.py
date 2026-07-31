from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)

from backend.training_service.domain.entities.checkpoint import (
    TrainingCheckpoint,
)

from backend.training_service.infrastructure.models.training_checkpoint_model import (
    TrainingCheckpointModel,
)

from backend.training_service.infrastructure.persistence_mappers.training_checkpoint_persistence_mapper import (
    TrainingCheckpointPersistenceMapper,
)


class SQLCheckpointRepository(CheckpointRepository):
    """
    SQLAlchemy implementation of CheckpointRepository.
    """

    def __init__(
        self,
        session: Session,
    ) -> None:

        self._session = session

    def save(
        self,
        checkpoint: TrainingCheckpoint,
    ) -> None:

        model = (
            TrainingCheckpointPersistenceMapper.to_model(
                checkpoint
            )
        )

        self._session.add(model)

        self._session.commit()

    def update(
        self,
        checkpoint: TrainingCheckpoint,
    ) -> None:

        model = self._session.get(
            TrainingCheckpointModel,
            checkpoint.id,
        )

        if model is None:
            raise ValueError(
                f"Checkpoint '{checkpoint.id}' not found."
            )

        updated = (
            TrainingCheckpointPersistenceMapper.to_model(
                checkpoint
            )
        )

        model.training_run_id = updated.training_run_id
        model.configuration_id = updated.configuration_id

        model.episode = updated.episode
        model.step = updated.step

        model.checkpoint_type = updated.checkpoint_type
        model.file_path = updated.file_path
        model.is_best = updated.is_best
        model.description = updated.description

        model.created_at = updated.created_at

        self._session.commit()

    def get_by_id(
        self,
        checkpoint_id: UUID,
    ) -> TrainingCheckpoint:

        model = self._session.get(
            TrainingCheckpointModel,
            checkpoint_id,
        )

        if model is None:
            raise ValueError(
                f"Checkpoint '{checkpoint_id}' not found."
            )

        return (
            TrainingCheckpointPersistenceMapper.to_entity(
                model
            )
        )

    def get_latest_for_run(
        self,
        training_run_id: UUID,
    ) -> TrainingCheckpoint | None:

        model = (
            self._session.query(
                TrainingCheckpointModel,
            )
            .filter(
                TrainingCheckpointModel.training_run_id
                == training_run_id
            )
            .order_by(
                TrainingCheckpointModel.created_at.desc()
            )
            .first()
        )

        if model is None:
            return None

        return (
            TrainingCheckpointPersistenceMapper.to_entity(
                model
            )
        )

    def get_best_for_run(
        self,
        training_run_id: UUID,
    ) -> TrainingCheckpoint | None:

        model = (
            self._session.query(
                TrainingCheckpointModel,
            )
            .filter(
                TrainingCheckpointModel.training_run_id
                == training_run_id,
                TrainingCheckpointModel.is_best.is_(True),
            )
            .order_by(
                TrainingCheckpointModel.created_at.desc()
            )
            .first()
        )

        if model is None:
            return None

        return (
            TrainingCheckpointPersistenceMapper.to_entity(
                model
            )
        )

    def list_by_run(
        self,
        training_run_id: UUID,
    ) -> list[TrainingCheckpoint]:

        models = (
            self._session.query(
                TrainingCheckpointModel,
            )
            .filter(
                TrainingCheckpointModel.training_run_id
                == training_run_id
            )
            .order_by(
                TrainingCheckpointModel.created_at.asc()
            )
            .all()
        )

        return [
            TrainingCheckpointPersistenceMapper.to_entity(
                model
            )
            for model in models
        ]

    def delete(
        self,
        checkpoint_id: UUID,
    ) -> None:

        model = self._session.get(
            TrainingCheckpointModel,
            checkpoint_id,
        )

        if model is None:
            raise ValueError(
                f"Checkpoint '{checkpoint_id}' not found."
            )

        self._session.delete(model)

        self._session.commit()