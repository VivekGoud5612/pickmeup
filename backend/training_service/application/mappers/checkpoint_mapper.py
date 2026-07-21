from __future__ import annotations

from training_service.domain.entities.training_checkpoint import (
    TrainingCheckpoint,
)

from training_service.application.dto.checkpoint.responses import (
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
            created_at=checkpoint.created_at,
            is_best=checkpoint.is_best,
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