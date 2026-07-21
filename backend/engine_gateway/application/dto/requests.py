from __future__ import annotations

from dataclasses import dataclass 
from uuid import UUID 
from pathlib import Path 



@dataclass(slots = True, frozen = True, kw_only = True)
class StartEngineTrainingRequest:
    """
    Request to start a new training session
    inside RL engine
    """
    configuration : TrainingConfiguration

    checkpoint_directory : Path 

    device : str | None = None 

    run_name : str 


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
class EvaluateCheckpointRequest:

    checkpoint_path : str    ## Previously it was checkpoint id.. changed it..

