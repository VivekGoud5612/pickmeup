from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from backend.training_service.domain.enums import TrainingAlgorithm
from backend.training_service.domain.value_objects import (
    CurriculumSettings,
    HyperParameters,
    RewardWeights,
)


@dataclass(slots = True, kw_only = True)
class TrainingConfiguration:
    """
    Represents the complete training configuration used by a
    training run.

    A configuration is versioned so experiments remain reproducible.
    """
    id : UUID = field(default_factory = uuid4)## Note that uuid4 is a functions and default factory calls uuid4() each time...

    family_id : UUID   ## Crated in applicaiton layer .. . . . . . .

    is_training : bool 
    
    version: int = 1

    name: str

    algorithm: TrainingAlgorithm

    hyperparameters: HyperParameters

    reward_weights: RewardWeights

    curriculum: CurriculumSettings

    description: str = ""

    created_at: datetime = field(
        default_factory = lambda: datetime.now(UTC)   ## as we want the current time for each time the object is created, so we use field but default factory needs a function so we wrap that datatime.now(UTC) into a zero argumented lambda function...
    )
