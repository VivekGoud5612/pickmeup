"""
Response DTOs returned by Training use cases.

Response DTOs expose only the information required by
clients and do not leak domain entities.
"""

from __future__ import annotations

from dataclasses import dataclass 
from datetime import datetime
from uuid import UUID 

from training_service.domain.enums import TrainingStatus 
from training_service.domain.value_objects import TrainingProgress 


@dataclass(slots = True, frozen = True, kw_only = True)
class TrainingSummaryResponse:
    """
    Summary response about a training run
    """

    id : UUID 

    name : str 

    status : TrainingStatus 

    progress : TrainingProgress 

    created_at : datetime 

    finished_at : datetime | None = None


@dataclass(slots = True, frozen = True, kw_only = True)
class TrainingCreatedResponse:
    """
    Response created after creating a training run
    """
    run : TrainingSummaryResponse

    message : str = "Training Created Successfully"


@dataclass(slots = True, frozen = True, kw_only = True)
class TrainingProgressResponse:

    id : UUID 
    name : str 
    status : TrainingStatus 
    progress : TrainingProgress


@dataclass(slots = True, frozen = True, kw_only = True)
class ListTrainingRunsResponse:

    runs : list[TrainingSummaryResponse]


@dataclass(slots = True, frozen = True, kw_only = True)
class DeleteTrainingResponse:

    message : str = "Training deleted Successfully"