from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from unittest.mock import MagicMock

import pytest

from backend.engine_gateway.infrastructure.clients.local_engine_client import LocalEngineClient

from backend.engine_gateway.application.dto.requests import (
    InitializeEngineTrainingRequest,
    StartEngineTrainingRequest,
    PauseEngineTrainingRequest,
    ResumeEngineTrainingRequest,
    StopEngineTrainingRequest,
    SaveEngineCheckpointRequest,
    DeleteEngineCheckpointRequest,
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

from engine.training.training_engine import TrainingEngine
from engine.utils.enums import EngineStatus


# -------------------------------------------------
# Fixtures
# -------------------------------------------------

@pytest.fixture
def dummy_configuration():

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
def engine():

    e = MagicMock(spec=TrainingEngine)

    e.state = EngineStatus.INITIALIZED

    e.global_step = 100

    e.checkpoint_directory = Path("checkpoints")

    return e


@pytest.fixture
def client(engine):

    return LocalEngineClient(engine)


# -------------------------------------------------
# Tests
# -------------------------------------------------

def test_initialize(client, engine, dummy_configuration):

    publisher = MagicMock()

    request = InitializeEngineTrainingRequest(
        run_id=uuid4(),
        configuration=dummy_configuration,
    )

    client.initialize(request, publisher)

    engine.initialize.assert_called_once()


def test_start(client, engine):

    engine.state = EngineStatus.RUNNING

    response = client.start(StartEngineTrainingRequest())

    engine.start.assert_called_once()

    assert response.status == EngineStatus.RUNNING


def test_pause(client, engine):

    engine.state = EngineStatus.PAUSED

    response = client.pause(PauseEngineTrainingRequest())

    engine.pause.assert_called_once()

    assert response.status == EngineStatus.PAUSED


def test_resume(client, engine):

    engine.state = EngineStatus.RUNNING

    response = client.resume(ResumeEngineTrainingRequest())

    engine.resume.assert_called_once()

    assert response.status == EngineStatus.RUNNING


def test_stop(client, engine):

    engine.state = EngineStatus.STOPPED

    response = client.stop(StopEngineTrainingRequest())

    engine.stop.assert_called_once()

    assert response.status == EngineStatus.STOPPED


def test_save_checkpoint(client, engine):

    with TemporaryDirectory() as tmp:

        engine.checkpoint_directory = Path(tmp)

        request = SaveEngineCheckpointRequest(
            checkpoint_name="checkpoint_1"
        )

        response = client.save_checkpoint(request)

        engine.save_checkpoint.assert_called_once()

        assert response.global_step == 100

        assert response.checkpoint_type.name == "MANUAL"

        assert response.checkpoint_path.parent.exists()


def test_delete_checkpoint(client):

    with TemporaryDirectory() as tmp:

        file = Path(tmp) / "checkpoint.pt"

        file.touch()

        request = DeleteEngineCheckpointRequest(
            checkpoint_path=file
        )

        client.delete_checkpoint(request)

        assert not file.exists()


def test_delete_checkpoint_raises_if_missing(client):

    with TemporaryDirectory() as tmp:

        file = Path(tmp) / "missing.pt"

        request = DeleteEngineCheckpointRequest(
            checkpoint_path=file
        )

        with pytest.raises(FileNotFoundError):
            client.delete_checkpoint(request)