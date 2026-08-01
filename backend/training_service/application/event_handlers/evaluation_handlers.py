from __future__ import annotations

from backend.engine_gateway.application.events.evaluation_events import (
    EvaluationStartedEvent,
    EvaluationCompletedEvent,
)

from backend.training_service.application.event_handlers.base import (
    EngineEventHandler,
)

from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)

from backend.training_service.application.repositories.evaluation_repository import (
    EvaluationRepository,
)

from backend.training_service.domain.entities.evaluation_result import (
    EvaluationResult,
)


class EvaluationStartedHandler(EngineEventHandler):

    def handle(
        self,
        event: EvaluationStartedEvent,
    ) -> None:
        """
        Currently no persistence is required.

        Reserved for future:
        - evaluation status
        - progress tracking
        - websocket notification
        """
        return


class EvaluationCompletedHandler(EngineEventHandler):

    def __init__(
        self,
        checkpoint_repo: CheckpointRepository,
        evaluation_repo: EvaluationRepository,
    ) -> None:

        self._checkpoint_repo = checkpoint_repo
        self._evaluation_repo = evaluation_repo

    def handle(
        self,
        event: EvaluationCompletedEvent,
    ) -> None:

        checkpoint = (
            self._checkpoint_repo.get_by_path(
                event.checkpoint_path,
            )
        )

        evaluation = EvaluationResult(
            checkpoint_id=checkpoint.id,
            metrics=event.metrics,
            evaluated_at=event.created_at,
        )

        self._evaluation_repo.save(
            evaluation,
        )