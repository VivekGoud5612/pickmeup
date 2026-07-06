from __future__ import annotations

from training_service.domain.entities.training_run import TrainingRun

from training_service.application.dto.training.responses import (
    TrainingSummaryResponse,
    TrainingCreatedResponse,
    TrainingProgressResponse,
    DeleteTrainingResponse,
)


class TrainingMapper:
    """
    Maps training entities to responses
    to be used in various commands and queries of 
    use cases...
    """

    @staticmethod
    def to_summary(
        training_run : TrainingRun,
    ) -> TrainingSummaryResponse:

        return TrainingSummaryResponse(
                id=training.id,
                name=training.name,
                status=training.status,
                progress=training.progress,
                started_at=training.started_at,
                finished_at=training.finished_at,
            )

    
    @staticmethod
    def to_created(
        training: TrainingRun,
    ) -> TrainingCreatedResponse:

        return TrainingCreatedResponse(
            summary=TrainingMapper.to_summary(training),
        )

    
    @staticmethod
    def to_progress(
        training: TrainingRun,
    ) -> TrainingProgressResponse:

        return TrainingProgressResponse(
            id=training.id,
            name=training.name,
            status=training.status,
            progress=training.progress,
        )

    
    @staticmethod
    def to_deleted(
        training : TrainingRun,
    ) -> DeleteTrainingResponse:

        return DeleteTrainingResponse(
            message=f"Checkpoint '{training.id}' deleted successfully."
        )