from __future__ import annotations

from backend.training_service.domain.entities.training_run import (
    TrainingRun,
)

from backend.training_service.domain.value_objects import (
    TrainingProgress,
)

from backend.training_service.infrastructure.models.training_run_model import (
    TrainingRunModel,
)


class TrainingRunPersistenceMapper:
    """
    Maps between the domain TrainingRun entity
    and the SQLAlchemy TrainingRunModel.
    """

    @staticmethod
    def to_model(
        training_run: TrainingRun,
    ) -> TrainingRunModel:

        return TrainingRunModel(

            id=training_run.id,

            name=training_run.name,

            algorithm=training_run.algorithm,

            status=training_run.status,

            step=training_run.progress.step if training_run.progress else 0,

            configuration_id=training_run.configuration_id,

            latest_checkpoint_id=training_run.latest_checkpoint_id,

            best_checkpoint_id=training_run.best_checkpoint_id,

            started_at=training_run.started_at,

            finished_at=training_run.finished_at,

            notes=training_run.notes,
        )

    @staticmethod
    def to_entity(
        model: TrainingRunModel,
    ) -> TrainingRun:

        return TrainingRun(

            id=model.id,

            name=model.name,

            algorithm=model.algorithm,

            status=model.status,

            progress=TrainingProgress(
                step=model.step,
            ),

            configuration_id=model.configuration_id,

            latest_checkpoint_id=model.latest_checkpoint_id,

            best_checkpoint_id=model.best_checkpoint_id,

            started_at=model.started_at,

            finished_at=model.finished_at,

            notes=model.notes,
        )