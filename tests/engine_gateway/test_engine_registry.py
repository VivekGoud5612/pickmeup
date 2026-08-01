from uuid import uuid4
from unittest.mock import MagicMock, patch

import pytest

from backend.engine_gateway.infrastructure.engine_registry import EngineRegistry

from backend.engine_gateway.application.dto.requests import (
    InitializeEngineTrainingRequest,
)

from backend.training_service.domain.entities.training_config import (
    TrainingConfiguration,
)

from backend.training_service.domain.enums import (
    TrainingAlgorithm,
)

from backend.training_service.domain.value_objects import (
    HyperParameters,
    RewardWeights,
    CurriculumSettings,
)


# -------------------------------------------------------
# Fixtures
# -------------------------------------------------------

@pytest.fixture
def configuration():

    return TrainingConfiguration(

        family_id=uuid4(),

        is_training=True,

        name="Unit Test",

        algorithm=TrainingAlgorithm.MAPPO,

        hyperparameters=HyperParameters(

            learning_rate=3e-4,
            total_timesteps=1000,
            num_envs=1,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            entropy_coeff=0.01,
            batch_size=64,
            rollout_length=32,
            ppo_epochs=4,
            max_grad_norm=0.5,
            checkpoint_save_interval=100,
        ),

        reward_weights=RewardWeights(

            dealer_damage=1,
            tank_damage=1,
            healer_healing=1,
            boss_damage=1,
            tank_block=1,
            healer_damage=1,
            boss_healing=1,

            death_penalty=-10,
            step_penalty=-0.01,
            victory_bonus=100,
            invalid_action_penalty=-1,
        ),

        curriculum=CurriculumSettings(

            boss_hp=100,

            spawn_radius=5,

            grid_size=10,

            max_steps=500,

            difficulty_level=1,
        ),
    )


@pytest.fixture
def initialize_request(configuration):

    return InitializeEngineTrainingRequest(

        run_id=uuid4(),

        configuration=configuration,
    )


@pytest.fixture
def repositories():

    training_repo = MagicMock()

    configuration_repo = MagicMock()

    checkpoint_repo = MagicMock()

    evaluation_repo = MagicMock()

    return (
        training_repo,
        configuration_repo,
        checkpoint_repo,
        evaluation_repo,
    )


# -------------------------------------------------------
# CREATE
# -------------------------------------------------------

@patch(
    "backend.engine_gateway.infrastructure.engine_registry.create_local_engine_event_publisher"
)

@patch(
    "backend.engine_gateway.infrastructure.engine_registry.LocalEngineClient"
)

@patch(
    "backend.engine_gateway.infrastructure.engine_registry.TrainingEngine"
)

def test_create_registers_engine(

    mock_engine,

    mock_client,

    mock_factory,

    initialize_request,

    repositories,

):

    (
        training_repo,
        configuration_repo,
        checkpoint_repo,
        evaluation_repo,
    ) = repositories

    registry = EngineRegistry()

    publisher = MagicMock()

    mock_factory.return_value = publisher

    engine = MagicMock()

    mock_engine.return_value = engine

    client = MagicMock()

    mock_client.return_value = client

    run_id = uuid4()

    returned = registry.create(

        training_run_id=run_id,

        request=initialize_request,

        training_repo=training_repo,

        configuration_repo=configuration_repo,

        checkpoint_repo=checkpoint_repo,

        evaluation_repo=evaluation_repo,
    )

    assert returned == client

    assert registry.exists(run_id)

    mock_factory.assert_called_once_with(

        training_repo=training_repo,

        configuration_repo=configuration_repo,

        checkpoint_repo=checkpoint_repo,

        evaluation_repo=evaluation_repo,
    )

    mock_engine.assert_called_once()

    mock_client.assert_called_once_with(
        engine=engine,
    )

    client.initialize.assert_called_once_with(

        request=initialize_request,

        publisher=publisher,
    )


# -------------------------------------------------------
# DUPLICATE CREATE
# -------------------------------------------------------

@patch(
    "backend.engine_gateway.infrastructure.engine_registry.create_local_engine_event_publisher"
)

@patch(
    "backend.engine_gateway.infrastructure.engine_registry.LocalEngineClient"
)

@patch(
    "backend.engine_gateway.infrastructure.engine_registry.TrainingEngine"
)

def test_duplicate_create_raises(

    mock_engine,

    mock_client,

    mock_factory,

    initialize_request,

    repositories,

):

    mock_factory.return_value = MagicMock()

    mock_engine.return_value = MagicMock()

    mock_client.return_value = MagicMock()

    registry = EngineRegistry()

    run_id = uuid4()

    registry.create(

        training_run_id=run_id,

        request=initialize_request,

        training_repo=repositories[0],

        configuration_repo=repositories[1],

        checkpoint_repo=repositories[2],

        evaluation_repo=repositories[3],
    )

    with pytest.raises(ValueError):

        registry.create(

            training_run_id=run_id,

            request=initialize_request,

            training_repo=repositories[0],

            configuration_repo=repositories[1],

            checkpoint_repo=repositories[2],

            evaluation_repo=repositories[3],
        )


# -------------------------------------------------------
# GET
# -------------------------------------------------------

def test_get_existing_client():

    registry = EngineRegistry()

    run_id = uuid4()

    client = MagicMock()

    registry._clients[run_id] = client

    returned = registry.get(run_id)

    assert returned == client


def test_get_missing_client():

    registry = EngineRegistry()

    with pytest.raises(ValueError):

        registry.get(uuid4())


# -------------------------------------------------------
# REMOVE
# -------------------------------------------------------

def test_remove():

    registry = EngineRegistry()

    run_id = uuid4()

    registry._clients[run_id] = MagicMock()

    registry.remove(run_id)

    assert registry.exists(run_id) is False


# -------------------------------------------------------
# EXISTS
# -------------------------------------------------------

def test_exists():

    registry = EngineRegistry()

    run_id = uuid4()

    registry._clients[run_id] = MagicMock()

    assert registry.exists(run_id)

    assert registry.exists(uuid4()) is False


# -------------------------------------------------------
# RUNNING RUNS
# -------------------------------------------------------

def test_running_training_runs():

    registry = EngineRegistry()

    r1 = uuid4()

    r2 = uuid4()

    registry._clients[r1] = MagicMock()

    registry._clients[r2] = MagicMock()

    runs = registry.running_training_runs()

    assert len(runs) == 2

    assert r1 in runs

    assert r2 in runs