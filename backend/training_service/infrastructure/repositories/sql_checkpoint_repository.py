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

        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        model.training_run_id = updated.training_run_id

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        model.episode = updated.episode
        model.step = updated.step

        # ----------------------------------------------------
        # Hyperparameters
        # ----------------------------------------------------

        model.learning_rate = updated.learning_rate
        model.total_timesteps = updated.total_timesteps
        model.num_envs = updated.num_envs
        model.gamma = updated.gamma
        model.gae_lambda = updated.gae_lambda
        model.clip_range = updated.clip_range
        model.entropy_coeff = updated.entropy_coeff
        model.batch_size = updated.batch_size
        model.rollout_length = updated.rollout_length
        model.ppo_epochs = updated.ppo_epochs
        model.max_grad_norm = updated.max_grad_norm
        model.checkpoint_save_interval = updated.checkpoint_save_interval

        # ----------------------------------------------------
        # Reward Weights
        # ----------------------------------------------------

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
        model.invalid_action_penalty = updated.invalid_action_penalty

        # ----------------------------------------------------
        # Curriculum
        # ----------------------------------------------------

        model.boss_hp = updated.boss_hp
        model.spawn_radius = updated.spawn_radius
        model.max_steps = updated.max_steps
        model.difficulty_level = updated.difficulty_level
        model.reward_scale = updated.reward_scale
        model.grid_size = updated.grid_size

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

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

    def get_by_path(
        self,
        checkpoint_path: Path,
        ) -> TrainingCheckpoint:

        model = (
            self._session.query(
                TrainingCheckpointModel,
            )
            .filter(
                TrainingCheckpointModel.file_path
                == str(checkpoint_path)
            )
            .first()
        )

        if model is None:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' not found."
            )

        return (
            TrainingCheckpointPersistenceMapper.to_entity(
                model,
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