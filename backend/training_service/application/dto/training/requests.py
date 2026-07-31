"""
Request DTOs for Training use cases.

These objects represent incoming requests from FastAPI,
CLI commands, MCP tools, or other application adapters.

DTOs contain no business logic.
"""

from __future__ import annotations 
from dataclasses import dataclass 
from uuid import UUID

from backend.training_service.domain.enums import TrainingAlgorithm, TrainingStatus
from backend.training_service.application.dto.training.nested_requests import HyperParametersRequest, RewardWeightsRequest, CurriculumSettingsRequest

@dataclass(slots = True, frozen = True, kw_only = True)  ## these are immutable objects 
class StartTrainingRequest:
    """
    Request to create and start a new training run 
    """

    name : str 

    configuration_name : str

    algorithm : TrainingAlgorithm 

    hyperparameters : HyperParametersRequest 

    reward_weights : RewardWeightsRequest

    curriculum_settings : CurriculumSettingsRequest  

    notes : str = ""


@dataclass(slots = True, frozen = True, kw_only = True)
class PauseTrainingRequest:
    """
    Request to pause an existing Training Run
    """
    
    training_run_id : UUID 


@dataclass(slots = True, frozen = True, kw_only = True)
class ResumeTrainingRequest:
    """
    Request to resume training
    """

    training_run_id : UUID 

@dataclass(slots = True, frozen = True, kw_only = True)
class StopTrainingRequest:
    """
    Request to stop training
    """
    training_run_id : UUID 

@dataclass(slots = True, frozen = True, kw_only = True)
class FailTrainingRequest:
    
    training_run_id : UUID 

@dataclass(slots = True, frozen = True, kw_only = True)
class DeleteTrainingRequest:

    training_run_id : UUID 

@dataclass(slots = True, frozen = True, kw_only = True)
class GetTrainingSummaryRequest:

    training_run_id : UUID 

@dataclass(slots = True, frozen = True, kw_only = True)
class GetTrainingProgressRequest:

    training_run_id : UUID  ## WE can get progress from that...

@dataclass(slots =True, frozen = True, kw_only = True)
class ListTrainingRunsRequest:

    status : TrainingStatus | None = None   ## So that I can request for specific conditioned lists, also 
    algorithm : TrainingAlgorithm | None = None   ## filtered for a specific algorithm...

    limit : int = 100
