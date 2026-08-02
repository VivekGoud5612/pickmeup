from __future__ import annotations

from backend.training_service.domain.entities.evaluation_result import (
    EvaluationResult,
)

from backend.training_service.domain.value_objects import (
    PerformanceMetrics,
)

from backend.training_service.infrastructure.models.evaluation_result_model import (
    EvaluationResultModel,
)


class EvaluationPersistenceMapper:
    """
    Maps between EvaluationResult domain entities
    and SQLAlchemy models.
    """

    @staticmethod
    def to_model(
        evaluation: EvaluationResult,
    ) -> EvaluationResultModel:

        return EvaluationResultModel(


            # Identity

            id=evaluation.id,

            checkpoint_id=evaluation.checkpoint_id,


            # Evaluation

            status=evaluation.status,

            notes=evaluation.notes,


            # Performance Metrics

            average_reward=evaluation.metrics.average_reward,

            actor_losses=evaluation.metrics.actor_losses,

            critic_losses=evaluation.metrics.critic_losses,

            entropy=evaluation.metrics.entropy,

            explained_variance=evaluation.metrics.explained_variance,

            win_rate=evaluation.metrics.win_rate,

            episode_length=evaluation.metrics.episode_length,


            # Timestamp

            created_at=evaluation.created_at,
        )

    @staticmethod
    def to_entity(
        model: EvaluationResultModel,
    ) -> EvaluationResult:

        return EvaluationResult(


            # Identity

            id=model.id,

            checkpoint_id=model.checkpoint_id,


            # Evaluation

            status=model.status,

            notes=model.notes,

            # Performance Metrics

            metrics=PerformanceMetrics(

                average_reward=model.average_reward,

                actor_losses=model.actor_losses,

                critic_losses=model.critic_losses,

                entropy=model.entropy,

                explained_variance=model.explained_variance,

                win_rate=model.win_rate,

                episode_length=model.episode_length,
            ),


            # Timestamp

            created_at=model.created_at,
        )