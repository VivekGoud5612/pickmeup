from __future__ import annotations 

from training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)

from training_service.application.dto.training.requests import (
    ListTrainingRunsRequest
)

from training_service.application.dto.training.responses import (
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
    ) -> ListTrainingRunsResponse

        training_runs = self._training_repo.list_all()

        return ListCheckpointsResponse(
            runs_summary = [
                TrainingMapper.to_summary(run)
                for run in training_runs
            ]
            )