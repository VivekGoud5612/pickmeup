from __future__ import annotations

from backend.training_service.domain.entities.training_run import TrainingRun

from backend.training_service.application.dto.training.responses import (
    TrainingSummaryResponse,
    TrainingCreatedResponse,
    TrainingProgressResponse,
    TrainingDeletedResponse,
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
                id=training_run.id,
                name=training_run.name,
                status=training_run.status,
                progress=training_run.progress,
                created_at=training_run.started_at,
                finished_at=training_run.finished_at,
            )

        
    @staticmethod
    def to_created(
        training_run: TrainingRun,
    ) -> TrainingCreatedResponse:

        return TrainingCreatedResponse(
            run_summary=TrainingMapper.to_summary(training_run),
        )

        
    @staticmethod
    def to_progress(
        training_run: TrainingRun,
    ) -> TrainingProgressResponse:

        return TrainingProgressResponse(
            id=training_run.id,
            name=training_run.name,
            status=training_run.status,
            progress=training_run.progress,
        )

        
    @staticmethod
    def to_deleted(
        training_run : TrainingRun,
    ) -> TrainingDeletedResponse:

        return TrainingDeletedResponse(
            message=f"Training '{training_run.id}' deleted successfully."
        )