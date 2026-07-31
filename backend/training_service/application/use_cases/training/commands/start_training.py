from __future__ import annotations

from uuid import uuid4

from backend.training_service.application.dto.training.requests import (
    StartTrainingRequest,
)
from backend.training_service.application.dto.training.responses import (
    TrainingCreatedResponse,
)

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from backend.training_service.application.repositories.training_configuration_repository import (
    TrainingConfigurationRepository,
)

from backend.training_service.application.mappers.training_mapper import (
    TrainingMapper,
)

from backend.training_service.domain.entities.training_run import (
    TrainingRun,
)

from backend.training_service.domain.entities.training_config import (
    TrainingConfiguration,
)

from backend.training_service.domain.enums import (
    TrainingStatus,
)

from backend.training_service.domain.value_objects import (
    HyperParameters,
    RewardWeights,
    CurriculumSettings,
    TrainingProgress,
)

from backend.engine_gateway.infrastructure.dummyengine_registry import (
    DummyEngineRegistry,
)


class StartTrainingUseCase:

    def __init__(
        self,
        training_repo: TrainingRunRepository,
        training_config_repo: TrainingConfigurationRepository,
        engine_registry: DummyEngineRegistry,
    ) -> None:

        self._training_repo = training_repo
        self._training_config_repo = training_config_repo
        self._engine_registry = engine_registry

    def execute(
        self,
        request: StartTrainingRequest,
    ) -> TrainingCreatedResponse:

        hyperparameters = self._create_hyperparameters(request)

        reward_weights = self._create_reward_weights(request)

        curriculum_settings = self._create_curriculum_settings(request)

        configuration = self._create_configuration(
            request,
            hyperparameters,
            reward_weights,
            curriculum_settings,
        )

        self._training_config_repo.save(configuration)

        training_run_id = uuid4()

        training_run = self._create_training_run(
            request,
            configuration,
            training_run_id,
        )

        training_run.start()

        self._training_repo.save(training_run)

        self._engine_registry.create(training_run_id)

        return TrainingMapper.to_created(
            training_run,
        )

    def _create_hyperparameters(
        self,
        request: StartTrainingRequest,
    ) -> HyperParameters:

        return HyperParameters(
            learning_rate=request.hyperparameters.learning_rate,
            num_envs=request.hyperparameters.num_envs,
            gamma=request.hyperparameters.gamma,
            gae_lambda=request.hyperparameters.gae_lambda,
            clip_range=request.hyperparameters.clip_range,
            entropy_coeff=request.hyperparameters.entropy_coeff,
            batch_size=request.hyperparameters.batch_size,
            rollout_length=request.hyperparameters.rollout_length,
            ppo_epochs=request.hyperparameters.ppo_epochs,
            max_grad_norm=request.hyperparameters.max_grad_norm,
            checkpoint_save_interval=request.hyperparameters.checkpoint_save_interval,
        )

    def _create_reward_weights(
        self,
        request: StartTrainingRequest,
    ) -> RewardWeights:

        return RewardWeights(
            dealer_damage=request.reward_weights.dealer_damage,
            tank_damage=request.reward_weights.tank_damage,
            healer_healing=request.reward_weights.healer_healing,
            boss_damage=request.reward_weights.boss_damage,
            tank_block=request.reward_weights.tank_block,
            healer_damage=request.reward_weights.healer_damage,
            boss_healing=request.reward_weights.boss_healing,
            death_penalty=request.reward_weights.death_penalty,
            step_penalty=request.reward_weights.step_penalty,
            victory_bonus=request.reward_weights.victory_bonus,
            invalid_action_penalty=request.reward_weights.invalid_action_penalty,
        )

    def _create_curriculum_settings(
        self,
        request: StartTrainingRequest,
    ) -> CurriculumSettings:

        return CurriculumSettings(
            boss_hp=request.curriculum_settings.boss_hp,
            spawn_radius=request.curriculum_settings.spawn_radius,
            max_steps=request.curriculum_settings.max_steps,
            difficulty_level=request.curriculum_settings.difficulty_level,
            reward_scale=request.curriculum_settings.reward_scale,
            grid_size=request.curriculum_settings.grid_size,
        )

    def _create_configuration(
        self,
        request: StartTrainingRequest,
        hyperparameters: HyperParameters,
        reward_weights: RewardWeights,
        curriculum_settings: CurriculumSettings,
    ) -> TrainingConfiguration:

        return TrainingConfiguration(
            family_id=uuid4(),
            is_training=True,
            version=1,
            name=request.configuration_name,
            algorithm=request.algorithm,
            hyperparameters=hyperparameters,
            reward_weights=reward_weights,
            curriculum=curriculum_settings,
            description=request.notes
        )

    def _create_training_run(
        self,
        request: StartTrainingRequest,
        configuration: TrainingConfiguration,
        training_run_id,
    ) -> TrainingRun:

        return TrainingRun(
            id=training_run_id,
            name=request.name,
            algorithm=request.algorithm,
            status=TrainingStatus.CREATED,
            progress=TrainingProgress(
                episode=0,
                step=0,
            ),
            configuration_id=configuration.id,
        )