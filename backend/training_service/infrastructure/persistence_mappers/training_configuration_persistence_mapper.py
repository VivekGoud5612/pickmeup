from __future__ import annotations

from backend.training_service.domain.entities.training_config import (
    TrainingConfiguration,
)

from backend.training_service.domain.value_objects import (
    HyperParameters,
    RewardWeights,
    CurriculumSettings,
)

from backend.training_service.infrastructure.models.training_configuration_model import (
    TrainingConfigurationModel,
)


class TrainingConfigurationPersistenceMapper:
    """
    Maps between TrainingConfiguration domain entities
    and SQLAlchemy models.
    """

    @staticmethod
    def to_model(
        configuration: TrainingConfiguration,
    ) -> TrainingConfigurationModel:

        return TrainingConfigurationModel(

            # -------------------------
            # Identity
            # -------------------------

            id=configuration.id,
            family_id=configuration.family_id,
            version=configuration.version,

            # -------------------------
            # Metadata
            # -------------------------

            name=configuration.name,
            algorithm=configuration.algorithm,
            description=configuration.description,
            created_at=configuration.created_at,

            # -------------------------
            # HyperParameters
            # -------------------------

            learning_rate=configuration.hyperparameters.learning_rate,
            total_timesteps=configuration.hyperparameters.total_timesteps,
            num_envs=configuration.hyperparameters.num_envs,
            gamma=configuration.hyperparameters.gamma,
            gae_lambda=configuration.hyperparameters.gae_lambda,
            clip_range=configuration.hyperparameters.clip_range,
            entropy_coeff=configuration.hyperparameters.entropy_coeff,
            batch_size=configuration.hyperparameters.batch_size,
            rollout_length=configuration.hyperparameters.rollout_length,
            ppo_epochs=configuration.hyperparameters.ppo_epochs,
            max_grad_norm=configuration.hyperparameters.max_grad_norm,
            checkpoint_save_interval=configuration.hyperparameters.checkpoint_save_interval,

            # -------------------------
            # Reward Weights
            # -------------------------

            dealer_damage=configuration.reward_weights.dealer_damage,
            tank_damage=configuration.reward_weights.tank_damage,
            healer_healing=configuration.reward_weights.healer_healing,
            boss_damage=configuration.reward_weights.boss_damage,
            tank_block=configuration.reward_weights.tank_block,
            healer_damage=configuration.reward_weights.healer_damage,
            boss_healing=configuration.reward_weights.boss_healing,
            death_penalty=configuration.reward_weights.death_penalty,
            step_penalty=configuration.reward_weights.step_penalty,
            victory_bonus=configuration.reward_weights.victory_bonus,
            invalid_action_penalty=configuration.reward_weights.invalid_action_penalty,

            # -------------------------
            # Curriculum
            # -------------------------

            boss_hp=configuration.curriculum.boss_hp,
            spawn_radius=configuration.curriculum.spawn_radius,
            max_steps=configuration.curriculum.max_steps,
            difficulty_level=configuration.curriculum.difficulty_level,
            reward_scale=configuration.curriculum.reward_scale,
            grid_size=configuration.curriculum.grid_size,
        )

    @staticmethod
    def to_entity(
        model: TrainingConfigurationModel,
    ) -> TrainingConfiguration:

        return TrainingConfiguration(

            # -------------------------
            # Identity
            # -------------------------

            id=model.id,
            family_id=model.family_id,
            version=model.version,

            # -------------------------
            # Metadata
            # -------------------------

            is_training=True,
            name=model.name,
            algorithm=model.algorithm,
            description=model.description,
            created_at=model.created_at,

            # -------------------------
            # HyperParameters
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

            curriculum=CurriculumSettings(
                boss_hp=model.boss_hp,
                spawn_radius=model.spawn_radius,
                max_steps=model.max_steps,
                difficulty_level=model.difficulty_level,
                reward_scale=model.reward_scale,
                grid_size=model.grid_size,
            ),
        )