from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from backend.training_service.domain.entities.training_config import (
    TrainingConfiguration,
)

from backend.training_service.domain.enums import CheckpointType


@dataclass(slots=True, frozen=True, kw_only=True)
class InitializeEngineTrainingRequest:
    """
    Initializes a new training session inside the engine.
    """
    run_id : UUID

    configuration: TrainingConfiguration

    checkpoint_directory: Path | None = None

    checkpoint_path : Path | None = None

    device: str | None = None



#Lifecycle
@dataclass(slots=True, frozen=True, kw_only=True)
class StartEngineTrainingRequest:
    """
    Starts a previously initialized engine.
    """
    pass


@dataclass(slots=True, frozen=True, kw_only=True)
class PauseEngineTrainingRequest:
    """
    Pauses the currently running training.
    """
    pass


@dataclass(slots=True, frozen=True, kw_only=True)
class ResumeEngineTrainingRequest:
    """
    Resumes training after either:
    - Pause
    - LoadCheckpoint
    """
    pass


@dataclass(slots=True, frozen=True, kw_only=True)
class StopEngineTrainingRequest:
    """
    Stops the current training session.
    """
    pass



#Checkpoints
@dataclass(slots=True, frozen=True, kw_only=True)
class SaveEngineCheckpointRequest:
    """
    Saves the current engine state.
    """

    checkpoint_name: str | None = None
    checkpoint_type: CheckpointType = CheckpointType.MANUAL



@dataclass(slots=True, frozen=True, kw_only=True)
class DeleteEngineCheckpointRequest:
    """
    Deletes a checkpoint from disk.
    """

    checkpoint_path: Path


@dataclass(slots=True, frozen=True, kw_only=True)
class EvaluateEngineCheckpointRequest:
    """
    Evaluates a checkpoint without resuming training.
    """

    checkpoint_path: Path


@dataclass(slots=True, frozen=True, kw_only=True)
class ReplayCheckpointRequest:
    """
    Runs inference using a checkpoint and streams
    gameplay to the frontend.
    """

    checkpoint_path: Path

    episodes: int = 1