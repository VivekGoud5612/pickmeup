from uuid import uuid4
from unittest.mock import MagicMock

from backend.training_service.application.use_cases.training.commands.start_training import (
    StartTrainingUseCase,
)

from backend.training_service.application.dto.training.requests import (
    StartTrainingRequest,
)

from backend.training_service.application.dto.training.nested_requests import (
    HyperParametersRequest,
    RewardWeightsRequest,
    CurriculumSettingsRequest,
)

from backend.training_service.domain.enums import (
    TrainingAlgorithm,
    TrainingStatus,
)

from backend.engine_gateway.application.dto.responses import (
    EngineTrainingStartedResponse,
)

from engine.utils.enums import EngineStatus

def test_start_training_creates_run_and_starts_engine():

    training_repo = MagicMock()
    config_repo = MagicMock()
    checkpoint_repo = MagicMock()
    evaluation_repo = MagicMock()

    registry = MagicMock()

    engine_client = MagicMock()

    registry.create.return_value = engine_client

    engine_client.start.return_value = EngineTrainingStartedResponse(
        status=EngineStatus.RUNNING
    )

    use_case = StartTrainingUseCase(
        training_repo,
        config_repo,
        checkpoint_repo,
        evaluation_repo,
        registry,
    )

    request = StartTrainingRequest(

        name="Run 1",

        configuration_name="Config",

        algorithm=TrainingAlgorithm.MAPPO,

        hyperparameters=HyperParametersRequest(
            learning_rate=3e-4,
            total_timesteps=10000,
            num_envs=1,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            entropy_coeff=0.01,
            batch_size=64,
            rollout_length=256,
            ppo_epochs=4,
            max_grad_norm=0.5,
            checkpoint_save_interval=1000,
        ),

        reward_weights=RewardWeightsRequest(
            dealer_damage=1,
            tank_damage=1,
            healer_healing=1,
            boss_damage=1,
            tank_block=1,
            healer_damage=1,
            boss_healing=1,
            death_penalty=-1,
            step_penalty=-0.01,
            victory_bonus=10,
            invalid_action_penalty=-1,
        ),

        curriculum_settings=CurriculumSettingsRequest(
            boss_hp=100,
            spawn_radius=5,
            grid_size=20,
            max_steps=1000,
            difficulty_level=1,
            reward_scale=1,
        ),

        notes="unit test",
    )

    response = use_case.execute(request)

    assert response.message == "Training Created Successfully"

    config_repo.save.assert_called_once()

    training_repo.save.assert_called_once()

    registry.create.assert_called_once()

    engine_client.start.assert_called_once()