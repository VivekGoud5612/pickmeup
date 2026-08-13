from __future__ import annotations

from dataclasses import dataclass 
from uuid import UUID 
from pathlib import Path 



@dataclass(slots = True, frozen = True, kw_only = True)
class InitializeEngineTrainingRequest:
    """
    Request to start a new training session
    inside RL engine
    """
    configuration : TrainingConfiguration

    checkpoint_directory : Path | None = None 

    device : str | None = None 

    run_name : str 


@dataclass(slots = True, frozen = True, kw_only = True)
class StartEngineTrainingRequest:
    ...


@dataclass(slots = True, frozen = True, kw_only = True)
class PauseEngineTrainingRequest:
    ...


@dataclass(slots = True, frozen = True, kw_only = True)
class ResumeEngineTrainingRequest:
    ...


@dataclass(slots = True, frozen = True, kw_only = True)
class StopEngineTrainingRequest:
    ...


@dataclass(slots = True, frozen = True, kw_only = True)
class SaveEngineCheckpointRequest: 

    checkpoint_name : str | None = None 

@dataclass(slots = True, frozen = True, kw_only = True)
class DeleteEngineCheckpointRequest:

    checkpoint_path : Path

@dataclass(slots = True, frozen = True, kw_only = True)
class EvaluateEngineCheckpointRequest:

    checkpoint_path : Path    ## Previously it was checkpoint id.. changed it..

