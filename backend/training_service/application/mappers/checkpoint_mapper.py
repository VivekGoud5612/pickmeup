from __future__ import annotations

from backend.training_service.domain.entities.checkpoint import (
    TrainingCheckpoint,
)

from backend.training_service.application.dto.checkpoint.responses import (
    CheckpointSummaryResponse,
    CheckpointCreatedResponse,
    CheckpointDeletedResponse,
)


class CheckpointMapper:
    """
    Maps checkpoint entities into response DTOs.
    """

    @staticmethod
    def to_summary(
        checkpoint: TrainingCheckpoint,
    ) -> CheckpointSummaryResponse:

        return CheckpointSummaryResponse(
            id=checkpoint.id,
            training_run_id=checkpoint.training_run_id,
            progress=checkpoint.progress,
            checkpoint_type=checkpoint.checkpoint_type,
            is_best=checkpoint.is_best,
            created_at=checkpoint.created_at,
        )

    @staticmethod
    def to_created(
        checkpoint: TrainingCheckpoint,
    ) -> CheckpointCreatedResponse:

        return CheckpointCreatedResponse(
            checkpoint_summary = CheckpointMapper.to_summary(
                checkpoint
            ),
        )

    
    @staticmethod 
    def to_deleted(
        checkpoint : TrainingCheckpoint,
    ) -> CheckpointDeletedResponse:

        return CheckpointDeletedResponse(
            message=f"Checkpoint '{checkpoint.id}' deleted successfully."
        )
