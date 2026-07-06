from __future__ import annotations 

from training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)
from training_service.application.repositories.evaluation_repository import (
    EvaluationRepository, 
)
from training_service.application.dto.evaluation.requests import (
    StartEvaluationRequest,
)
from training_service.application.dto.evaluation.responses import (
    EvaluationCreatedResponse,
)
from training_service.domain.entities.evaluation import (
    EvaluationResult,
)
from training_service.domain.value_objects import (
    PerformanceMetrics,
)
from training_service.application.mappers.evaluation_mapper import (
    EvaluationMapper,
)


class StartEvaluationUseCase:

    def __init__(
        self,
        checkpoint_repo : CheckpointRepository,
        evaluation_repo : EvaluationRepository,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo
        self._evaluation_repo = evaluation_repo 

    
    def exectue(self, request : StartEvaluationRequest) -> EvaluationCreatedResponse:

        checkpoint = self._checkpoint_repo.get_by_id(request.checkpoint_id)

        performance = self._create_performance_metrics(request)

        evaluation = self._create_evaluation(
            checkpoint,
            performance,
            request,
        )

        evaluation.start()

        self._evaluation_repo.save(evaluation)

        return EvaluationMapper.to_created(evaluation)
    
    def _create_performance_metrics(self, request : StartEvaluationRequest) -> PerformanceMetrics:

        return PerformanceMetrics(
                average_reward = request.average_reward,
                actor_losses = request.actor_losses,
                critic_losses = request.critic_losses,
                entropy = request.entropy,
                explained_variance = request.explained_variance,
                win_rate = request.win_rate,
                episode_length = request.episode_length,
        )

    def _create_evaluation(self, checkpoint : TrainingCheckpoint, performance : PerformanceMetrics, request : StartEvaluationRequest) -> EvaluationResult:

        return EvaluationResult(
            checkpoint_id = checkpoint.id,
            metrics = performance,
            num_episodes = request.evaluation_episodes,
            notes = request.notes
        )
        