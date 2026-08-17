from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from backend.contracts.engine.models.training_configuration import TrainingConfiguration



@dataclass(slots = True, frozen = True, kw_only = True)
class InitializeEngineTrainingRequest:
    """
    Request to start a new training session
    inside RL engine
    """
    configuration: TrainingConfiguration
    run_id: UUID
    checkpoint_directory: Path | None = None
    device: str | None = None
    run_name: str


@dataclass(slots = True, frozen = True, kw_only = True)
class StartEngineTrainingRequest:
    run_id : UUID


@dataclass(slots = True, frozen = True, kw_only = True)
class PauseEngineTrainingRequest:
    run_id : UUID


@dataclass(slots = True, frozen = True, kw_only = True)
class ResumeEngineTrainingRequest:
    run_id : UUID


@dataclass(slots = True, frozen = True, kw_only = True)
class StopEngineTrainingRequest:
    run_id : UUID


@dataclass(slots = True, frozen = True, kw_only = True)
class SaveEngineCheckpointRequest: 
    checkpoint_name: str | None = None

@dataclass(slots = True, frozen = True, kw_only = True)
class DeleteEngineCheckpointRequest:
    checkpoint_path: Path

@dataclass(slots = True, frozen = True, kw_only = True)
class EvaluateEngineCheckpointRequest:
    checkpoint_path: Path


@dataclass(slots=True, frozen=True, kw_only=True)
class LoadEngineCheckpointRequest:
    checkpoint_path: Path
