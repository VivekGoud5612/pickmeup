from __future__ import annotations 

from dataclasses import dataclass, field 
from datetime import datetime, UTC 
from pathlib import Path 
from uuid import UUID, uuid4

from backend.training_service.domain.enums import CheckpointType
from backend.training_service.domain.value_objects import (
    CurriculumSettings,
    HyperParameters,
    RewardWeights,
)


@dataclass(slots=True, kw_only=True)
class TrainingCheckpoint:
    """
    Represents a saved snapshot of a training run.

    A checkpoint captures the model state at a particular point
    during training and can later be restored or evaluated.

    So each time a new checkpoint is created, this object gets changed and some methods 
    help in further tasks 
    """

    training_run_id : UUID 

    progress : TrainingProgress 

    checkpoint_type : CheckpointType 

    file_path : Path   ## So that in future we can check if path exists or any other Path Object method..

    hyperparameters : HyperParameters 

    reward_weights : RewardWeights   ## So that we can store the set of these parameters at checkpoint time as well..

    curriculum_settings : CurriculumSettings

    id : UUID = field(default_factory = uuid4)

    is_best : bool = False 

    created_at : datetime = field(default_factory = lambda: datetime.now(UTC))

    description : str = ""

    def mark_as_best(self) -> None:
        """
        Marks this checkpoint as the best performing checkpoint.
        """

        self.is_best = True

    def rename(self, description: str) -> None:
        """
        Update checkpoint description.
        """
        
        self.description = description