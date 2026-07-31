from __future__ import annotations 

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)

from backend.training_service.application.dto.training.requests import (
    ListTrainingRunsRequest
)

from backend.training_service.application.mappers.training_mapper import (
    TrainingMapper
)

from backend.training_service.application.dto.training.responses import (
    ListTrainingRunsResponse
)

class ListTrainingRunsUseCase:

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ):
        self._training_repo = training_repo

    def execute(
        self,
        request : ListTrainingRunsRequest,
    ) -> ListTrainingRunsResponse:

        training_runs = self._training_repo.list_all()

        return ListTrainingRunsResponse(
            runs_summary = [
                TrainingMapper.to_summary(run)
                for run in training_runs
            ]
            )