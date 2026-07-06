from  __future__ import annotations

from training_service.application.dto.training.requests import (
    StartTrainingRequest,
)
from training_service.application.dto.training.responses import (
    TrainingCreatedResponse,
)
from training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from training_service.application.repositories.training_configuration_repository import (
    TrainingConfigurationRepository,
)

from training_service.domain.entities import (
    TrainingRun,
    TrainingConfiguration,
)
from training_service.domain.value_objects import (
    HyperParameters,
    RewardWeights,
    CurriculumSettings,
)
from training_service.application.mappers.training_mapper import (
    TrainingMapper,
)


class StartTrainingUseCase:

    def __init__(
        self,
        _training_repo : TrainingRunRepository,
        _training_config_repo : TrainingConfigurationRepository,
    ) -> None:
        self._training_repo = _training_repo 
        self._training_config_repo = _training_config_repo


    def execute(
        self,
        request : StartTrainingRequest,
    ) -> TrainingCreatedResponse:

        hyperparameters = self._create_hyperparameters(request)

        reward_weights = self._create_reward_weights(request)

        curriculum_settings = self._create_curriculum_settings(request)

        configuration = self._create_configuration(request, hyperparameters, reward_weights, curriculum_settings)

        self._training_config_repo.save(configuration)

        training_run = self._create_training_run(request)
        training_run.start()  ## Start the training and we reutrn the response DTO after saving...

        self._training_repo.save(training_run)

        return TrainingMapper.to_created(training_run)

    
    def _create_hyperparameters(self, request : StartTrainingRequest) -> HyperParameters:

        return HyperParameters(
            learning_rate = request.hyperparameters.learning_rate,

            gamma = request.hyperparameters.gamma,
            gae_lambda = request.hyperparameters.gae_lambda,

            clip_range = request.hyperparameters.clip_range ,
            entropy_coeff = request.hyperparameters.entropy_coef,

            batch_size = request.hyperparameters.batch_size,
            rollout_length = request.hyperparameters.rollout_length,
            ppo_epochs = request.hyperparameters.ppo_epochs ,

            max_grad_norm = request.hyperparameters.max_grad_norm, 
        )
    
    def _create_reward_weights(self, request : StartTrainingRequest) -> RewardWeights:

        return RewardWeights(
            dealer_damage = request.reward_weights.dealer_damage, 
            tank_damage = request.reward_weights.tank_damage ,
            healer_healing = request.reward_weights.healer_healing,
            boss_damage = request.reward_weights.boss_damage,

            tank_block = request.reward_weights.tank_block,
            healer_damage = request.reward_weights.healer_damage,
            boss_healing = request.reward_weights.boss_healing,

            death_penalty = request.reward_weights.death_penalty,
            step_penalty = request.reward_weights.step_penalty,

            victory_bonus = request.reward_weights.victory_bonus,

            invalid_action_penalty = request.reward_weights.invalid_action_penalty,
        )

    
    def _create_curriculum_settings(self, request : StartTrainingRequest) -> CurriculumSettings:

        return CurriculumSettings(
            boss_hp = request.curriculum_settings.boss_hp,
            spawn_radius = request.curriculum_settings.spawn_radius,  
            max_episode_steps = request.curriculum_settings.max_episode_steps,

            difficulty_level = request.curriculum_settings.difficulty_level,

            reward_scale = request.curriculum_settings.reward_scale,
        )
    

    def _create_configuration(
        self, 
        request : StartTrainingRequest,
        hyperparameters : HyperParameters,
        reward_weights : RewardWeights,
        curriculum_settings : CurriculumSettings,
        ) -> TrainingConfiguration:

        return TrainingConfiguration(

            name=request.configuration_name,   ## Id will be created with this object creation...
            family_id = uuid4(),  ## Still deciding if I write this as field or get this from the request.. 
            version = 1,

            hyperparameters = hyperparameters,
            reward_weights = reward_weights,
            curriculum = curriculum_settings,
        )

    
    def _create_training_run(self, request : StartTrainingRequest, configuration : TrainingConfiguration) -> TrainingRun:
        """
        Create a training run and return that, writing this string 
        just for the sake of writing it...
        """

        return TrainingRun(

            name=request.name,
            algorithm=request.algorithm,   ## This is just the creation of the training run.. when we run this then we can intitialize progress...
            progress = TrainingProgress(
                episode = 0,
                step = 0,
            )
            configuration_id=configuration.id,
        )

    
    def _create_response_dto(self, training_run : TrainingRun) -> TrainingCreatedResponse:
        
        summary = TrainingSummaryResponse(
            id = training_run.id,
            name = training_run.name,
            status = training_run.status,
            progress = training_run.progress,
            created_at = training_run.created_at,
        )
        return TrainingCreatedResponse(
            run = summary,
        )
