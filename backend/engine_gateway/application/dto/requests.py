from __future__ import annotations

from dataclasses import dataclass 
from uuid import UUID 
from pathlib import Path 

from training_service.domain.enums import CheckpointType


@dataclass(slots = True, frozen = True, kw_only = True)
class StartEngineTrainingRequest:
    """
    Request to start a new training session
    inside RL engine
    """
    training_run_id : UUID 

    configuration : TrainingConfiguration

    checkpoint_directory ; Path 

    device : str 

    num_workers : str 


@dataclass(slots = True, frozen = True, kw_only = True)
class PauseEngineTrainingRequest:

    training_run_id : UUID 


@dataclass(slots = True, frozen = True, kw_only = True)
class ResumeEngineTrainingRequest:

    training_run_id : UUID 


@dataclass(slots = True, frozen = True, kw_only = True)
class StopEngineTrainingRequest:

    training_run_id : UUID 


@dataclass(slots = True, frozen = True, kw_only = True)
class SaveEngineCheckpointRequest:

    training_run_id : UUID 

    checkpoint_type : CheckpointType  


@dataclass(slots = True, frozen = True, kw_only = True)
class EvaluateCheckpointRequest:

    checkpoint_id : UUID 

    num_episodes : int = 100

