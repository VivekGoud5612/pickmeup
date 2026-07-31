from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from backend.training_service.application.repositories.training_configuration_repository import (
    TrainingConfigurationRepository,
)

from backend.training_service.domain.entities.training_config import (
    TrainingConfiguration,
)

from backend.training_service.infrastructure.models.training_configuration_model import (
    TrainingConfigurationModel,
)

from backend.training_service.infrastructure.persistence_mappers.training_configuration_persistence_mapper import (
    TrainingConfigurationPersistenceMapper,
)


class SQLTrainingConfigurationRepository(
    TrainingConfigurationRepository,
):
    """
    SQLAlchemy implementation of TrainingConfigurationRepository.
    """

    def __init__(
        self,
        session: Session,
    ) -> None:

        self._session = session

    def save(
        self,
        configuration: TrainingConfiguration,
    ) -> None:

        model = (
            TrainingConfigurationPersistenceMapper.to_model(
                configuration
            )
        )

        self._session.add(model)

        self._session.commit()

    def update(
        self,
        configuration: TrainingConfiguration,
    ) -> None:

        model = self._session.get(
            TrainingConfigurationModel,
            configuration.id,
        )

        if model is None:
            raise ValueError(
                f"Configuration '{configuration.id}' not found."
            )

        updated = (
            TrainingConfigurationPersistenceMapper.to_model(
                configuration
            )
        )

        # -------------------------
        # Metadata
        # -------------------------

        model.family_id = updated.family_id
        model.version = updated.version

        model.name = updated.name
        model.algorithm = updated.algorithm
        model.description = updated.description
        model.created_at = updated.created_at

        # -------------------------
        # HyperParameters
        # -------------------------

        model.learning_rate = updated.learning_rate
        model.num_envs = updated.num_envs
        model.gamma = updated.gamma
        model.gae_lambda = updated.gae_lambda
        model.clip_range = updated.clip_range
        model.entropy_coeff = updated.entropy_coeff
        model.batch_size = updated.batch_size
        model.rollout_length = updated.rollout_length
        model.ppo_epochs = updated.ppo_epochs
        model.max_grad_norm = updated.max_grad_norm
        model.checkpoint_save_interval = (
            updated.checkpoint_save_interval
        )

        # -------------------------
        # Reward Weights
        # -------------------------

        model.dealer_damage = updated.dealer_damage
        model.tank_damage = updated.tank_damage
        model.healer_healing = updated.healer_healing
        model.boss_damage = updated.boss_damage

        model.tank_block = updated.tank_block
        model.healer_damage = updated.healer_damage
        model.boss_healing = updated.boss_healing

        model.death_penalty = updated.death_penalty
        model.step_penalty = updated.step_penalty
        model.victory_bonus = updated.victory_bonus
        model.invalid_action_penalty = (
            updated.invalid_action_penalty
        )

        # -------------------------
        # Curriculum
        # -------------------------

        model.boss_hp = updated.boss_hp
        model.spawn_radius = updated.spawn_radius
        model.max_steps = updated.max_steps
        model.difficulty_level = updated.difficulty_level
        model.reward_scale = updated.reward_scale
        model.grid_size = updated.grid_size

        self._session.commit()

    def get_by_id(
        self,
        configuration_id: UUID,
    ) -> TrainingConfiguration:

        model = self._session.get(
            TrainingConfigurationModel,
            configuration_id,
        )

        if model is None:
            raise ValueError(
                f"Configuration '{configuration_id}' not found."
            )

        return (
            TrainingConfigurationPersistenceMapper.to_entity(
                model
            )
        )

    def delete(
        self,
        configuration_id: UUID,
    ) -> None:

        model = self._session.get(
            TrainingConfigurationModel,
            configuration_id,
        )

        if model is None:
            raise ValueError(
                f"Configuration '{configuration_id}' not found."
            )

        self._session.delete(model)

        self._session.commit()

    def exists(
        self,
        configuration_id: UUID,
    ) -> bool:

        return (
            self._session.get(
                TrainingConfigurationModel,
                configuration_id,
            )
            is not None
        )

    def list_all(
        self,
    ) -> list[TrainingConfiguration]:

        models = (
            self._session.query(
                TrainingConfigurationModel,
            )
            .all()
        )

        return [
            TrainingConfigurationPersistenceMapper.to_entity(
                model
            )
            for model in models
        ]