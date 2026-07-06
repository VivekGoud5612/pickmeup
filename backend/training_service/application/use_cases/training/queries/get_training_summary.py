from __future__ import annotations

from training_service.application.dto.training.requests import (
    GetTrainingSummaryRequest,
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


class GetTrainingSummaryUseCase:
    """
    Retrieve the current summary of a training run
    """

    def __init__(self, training_repo : TrainingRunRepository) -> None:

        self.training_repo = training_repo

    
    def execute(self, request : GetTrainingSummaryRequest) -> TrainingSummaryResponse:

        training_run = self._training_repo.get_by_id(request.training_run_id)

        return TrainingMapper.to_summary(training_run)