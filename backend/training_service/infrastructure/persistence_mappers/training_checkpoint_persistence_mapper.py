from __future__ import annotations

from pathlib import Path

from backend.training_service.domain.entities.checkpoint import (
    TrainingCheckpoint,
)

from backend.training_service.domain.value_objects import (
    TrainingProgress,
    HyperParameters,
    RewardWeights,
    CurriculumSettings,
)

from backend.training_service.infrastructure.models.training_checkpoint_model import (
    TrainingCheckpointModel,
)


class TrainingCheckpointPersistenceMapper:

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

            # -------------------------
            # Progress
            # -------------------------

            step=checkpoint.progress.step,

            # -------------------------
            # Hyperparameters
            # -------------------------

            learning_rate=checkpoint.hyperparameters.learning_rate,
            total_timesteps=checkpoint.hyperparameters.total_timesteps,
            num_envs=checkpoint.hyperparameters.num_envs,
            gamma=checkpoint.hyperparameters.gamma,
            gae_lambda=checkpoint.hyperparameters.gae_lambda,
            clip_range=checkpoint.hyperparameters.clip_range,
            entropy_coeff=checkpoint.hyperparameters.entropy_coeff,
            batch_size=checkpoint.hyperparameters.batch_size,
            rollout_length=checkpoint.hyperparameters.rollout_length,
            ppo_epochs=checkpoint.hyperparameters.ppo_epochs,
            max_grad_norm=checkpoint.hyperparameters.max_grad_norm,
            checkpoint_save_interval=checkpoint.hyperparameters.checkpoint_save_interval,

            # -------------------------
            # Reward Weights
            # -------------------------

            dealer_damage=checkpoint.reward_weights.dealer_damage,
            tank_damage=checkpoint.reward_weights.tank_damage,
            healer_healing=checkpoint.reward_weights.healer_healing,
            boss_damage=checkpoint.reward_weights.boss_damage,
            tank_block=checkpoint.reward_weights.tank_block,
            healer_damage=checkpoint.reward_weights.healer_damage,
            boss_healing=checkpoint.reward_weights.boss_healing,
            death_penalty=checkpoint.reward_weights.death_penalty,
            step_penalty=checkpoint.reward_weights.step_penalty,
            victory_bonus=checkpoint.reward_weights.victory_bonus,
            invalid_action_penalty=checkpoint.reward_weights.invalid_action_penalty,

            # -------------------------
            # Curriculum
            # -------------------------

            boss_hp=checkpoint.curriculum_settings.boss_hp,
            spawn_radius=checkpoint.curriculum_settings.spawn_radius,
            max_steps=checkpoint.curriculum_settings.max_steps,
            difficulty_level=checkpoint.curriculum_settings.difficulty_level,
            reward_scale=checkpoint.curriculum_settings.reward_scale,
            grid_size=checkpoint.curriculum_settings.grid_size,

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

            # -------------------------
            # Progress
            # -------------------------

            progress=TrainingProgress(
                step=model.step,
            ),

            # -------------------------
            # Hyperparameters
            # -------------------------

            hyperparameters=HyperParameters(
                learning_rate=model.learning_rate,
                total_timesteps=model.total_timesteps,
                num_envs=model.num_envs,
                gamma=model.gamma,
                gae_lambda=model.gae_lambda,
                clip_range=model.clip_range,
                entropy_coeff=model.entropy_coeff,
                batch_size=model.batch_size,
                rollout_length=model.rollout_length,
                ppo_epochs=model.ppo_epochs,
                max_grad_norm=model.max_grad_norm,
                checkpoint_save_interval=model.checkpoint_save_interval,
            ),

            # -------------------------
            # Reward Weights
            # -------------------------

            reward_weights=RewardWeights(
                dealer_damage=model.dealer_damage,
                tank_damage=model.tank_damage,
                healer_healing=model.healer_healing,
                boss_damage=model.boss_damage,
                tank_block=model.tank_block,
                healer_damage=model.healer_damage,
                boss_healing=model.boss_healing,
                death_penalty=model.death_penalty,
                step_penalty=model.step_penalty,
                victory_bonus=model.victory_bonus,
                invalid_action_penalty=model.invalid_action_penalty,
            ),

            # -------------------------
            # Curriculum
            # -------------------------

            curriculum_settings=CurriculumSettings(
                boss_hp=model.boss_hp,
                spawn_radius=model.spawn_radius,
                max_steps=model.max_steps,
                difficulty_level=model.difficulty_level,
                reward_scale=model.reward_scale,
                grid_size=model.grid_size,
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