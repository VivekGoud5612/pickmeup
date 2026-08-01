from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)

from backend.training_service.domain.entities.training_run import (
    TrainingRun,
)

from backend.training_service.infrastructure.models.training_run_model import (
    TrainingRunModel,
)

from backend.training_service.infrastructure.persistence_mappers.training_run_persistence_mapper import (
    TrainingRunPersistenceMapper,
)


class SQLTrainingRunRepository(TrainingRunRepository):
    """
    SQLAlchemy implementation of TrainingRunRepository.
    """

    def __init__(
        self,
        session: Session,
    ) -> None:

        self._session = session

    def save(
        self,
        training_run: TrainingRun,
    ) -> None:

        model = TrainingRunPersistenceMapper.to_model(
            training_run
        )

        self._session.add(model)

        self._session.commit()

    def update(
        self,
        training_run: TrainingRun,
    ) -> None:

        model = self._session.get(
            TrainingRunModel,
            training_run.id,
        )

        if model is None:
            raise ValueError(
                f"Training run '{training_run.id}' not found."
            )

        updated = TrainingRunPersistenceMapper.to_model(
            training_run
        )

        model.name = updated.name
        model.algorithm = updated.algorithm
        model.status = updated.status
        model.step = updated.step
        model.configuration_id = updated.configuration_id
        model.latest_checkpoint_id = updated.latest_checkpoint_id
        model.best_checkpoint_id = updated.best_checkpoint_id
        model.started_at = updated.started_at
        model.finished_at = updated.finished_at
        model.notes = updated.notes

        self._session.commit()

    def get_by_id(
        self,
        training_run_id: UUID,
    ) -> TrainingRun:

        model = self._session.get(
            TrainingRunModel,
            training_run_id,
        )

        if model is None:
            raise ValueError(
                f"Training run '{training_run_id}' not found."
            )

        return TrainingRunPersistenceMapper.to_entity(
            model
        )

    def delete(
        self,
        training_run_id: UUID,
    ) -> None:

        model = self._session.get(
            TrainingRunModel,
            training_run_id,
        )

        if model is None:
            raise ValueError(
                f"Training run '{training_run_id}' not found."
            )

        self._session.delete(model)

        self._session.commit()

    def exists(
        self,
        training_run_id: UUID,
    ) -> bool:

        return (
            self._session.get(
                TrainingRunModel,
                training_run_id,
            )
            is not None
        )

    def list_all(
        self,
    ) -> list[TrainingRun]:

        models = (
            self._session.query(
                TrainingRunModel,
            )
            .all()
        )

        return [
            TrainingRunPersistenceMapper.to_entity(
                model
            )
            for model in models
        ]