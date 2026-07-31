from __future__ import annotations 

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)

from backend.training_service.applicaiton.dto.training.requests import (
    GetTrainingProgressRequest,
)

from backend.training_service.applicaiton.dto.training.responses import (
    TrainingProgressResponse,
)
from backend.training_service.application.mappers.training_mapper import (
    TrainingMapper,
)


class GetTrainingProgressUseCase:

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ):
        self._training_repo = training_repo

    def execute(
        self,
        request: GetTrainingProgressRequest,
    ) -> TrainingProgressResponse:

        training_run = self._training_repo.get_by_id(
            request.training_run_id
        )

        return TrainingMapper.to_progress(training_run)