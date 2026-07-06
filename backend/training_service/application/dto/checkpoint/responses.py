from __future__ import annotations
from uuid import UUID 
from dataclasses import dataclass 

from training_service.domain.enums import EvaluationStatus

@dataclass(slots=True, frozen=True, kw_only=True)
class CheckpointSummaryResponse:

    id: UUID

    training_run_id: UUID

    progress : TrainingProgress 

    created_at: datetime    # No file system path, as we can get that any time with ID... And this should be DTO so no file paths 


@dataclass(slots=True, frozen=True, kw_only=True)
class CheckpointCreatedResponse:  ## After saving I guess ... 

    checkpoint_summary: CheckpointSummaryResponse

    message: str = "Checkpoint saved successfully."

@dataclass(slots=True, frozen=True, kw_only=True)
class DeleteCheckpointResponse:

    message: str = "Checkpoint deleted successfully."

@dataclass(slots=True, frozen=True, kw_only=True)
class ListCheckpointsResponse:

    checkpoints: list[CheckpointSummaryResponse]