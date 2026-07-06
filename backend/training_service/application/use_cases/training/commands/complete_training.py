from __future__ import annotations

from training_service.application.dto.training.requests import (
    CompleteTrainingRequest,
)
from training_service.application.dto.training.responses import (
    TrainingSummaryResponse,
)
from training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)
from training_service.application.mappers.training_mapper import (
    TrainingMapper,
)


class CompleteTrainingUseCase:

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ) -> None:
        self._training_repo = training_repo

    def execute(
        self,
        request: CompleteTrainingRequest,
    ) -> TrainingSummaryResponse:

        training_run = self._training_repo.get_by_id(
            request.training_run_id
        )

        training_run.complete()

        self._training_repo.update(training_run)

        return TrainingMapper.to_summary(training_run)

    
