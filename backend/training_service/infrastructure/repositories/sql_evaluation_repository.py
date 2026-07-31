from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from backend.training_service.application.repositories.evaluation_repository import (
    EvaluationRepository,
)

from backend.training_service.domain.entities.evaluation_result import (
    EvaluationResult,
)

from backend.training_service.infrastructure.models.evaluation_result_model import (
    EvaluationResultModel,
)

from backend.training_service.infrastructure.persistence_mappers.evaluation_result_persistence_mapper import (
    EvaluationPersistenceMapper,
)


class SQLEvaluationRepository(EvaluationRepository):
    """
    SQLAlchemy implementation of EvaluationRepository.
    """

    def __init__(
        self,
        session: Session,
    ) -> None:

        self._session = session

    def save(
        self,
        evaluation: EvaluationResult,
    ) -> None:

        model = EvaluationPersistenceMapper.to_model(
            evaluation
        )

        self._session.add(model)

        self._session.commit()

    def update(
        self,
        evaluation: EvaluationResult,
    ) -> None:

        model = self._session.get(
            EvaluationResultModel,
            evaluation.id,
        )

        if model is None:
            raise ValueError(
                f"Evaluation '{evaluation.id}' not found."
            )

        updated = EvaluationPersistenceMapper.to_model(
            evaluation
        )

        model.checkpoint_id = updated.checkpoint_id

        model.status = updated.status

        model.num_episodes = updated.num_episodes

        model.notes = updated.notes

        model.average_reward = updated.average_reward
        model.actor_losses = updated.actor_losses
        model.critic_losses = updated.critic_losses
        model.entropy = updated.entropy
        model.explained_variance = updated.explained_variance
        model.win_rate = updated.win_rate
        model.episode_length = updated.episode_length

        model.created_at = updated.created_at
        model.finished_at = updated.finished_at

        self._session.commit()

    def get_by_id(
        self,
        evaluation_id: UUID,
    ) -> EvaluationResult:

        model = self._session.get(
            EvaluationResultModel,
            evaluation_id,
        )

        if model is None:
            raise ValueError(
                f"Evaluation '{evaluation_id}' not found."
            )

        return EvaluationPersistenceMapper.to_entity(
            model
        )

    def list_by_checkpoint(
        self,
        checkpoint_id: UUID,
    ) -> list[EvaluationResult]:

        models = (
            self._session.query(
                EvaluationResultModel,
            )
            .filter(
                EvaluationResultModel.checkpoint_id
                == checkpoint_id
            )
            .order_by(
                EvaluationResultModel.created_at.desc()
            )
            .all()
        )

        return [
            EvaluationPersistenceMapper.to_entity(
                model
            )
            for model in models
        ]

    def list_all(
        self,
    ) -> list[EvaluationResult]:

        models = (
            self._session.query(
                EvaluationResultModel,
            )
            .order_by(
                EvaluationResultModel.created_at.desc()
            )
            .all()
        )

        return [
            EvaluationPersistenceMapper.to_entity(
                model
            )
            for model in models
        ]

    def delete(
        self,
        evaluation_id: UUID,
    ) -> None:

        model = self._session.get(
            EvaluationResultModel,
            evaluation_id,
        )

        if model is None:
            raise ValueError(
                f"Evaluation '{evaluation_id}' not found."
            )

        self._session.delete(model)

        self._session.commit()