from __future__ import annotations

from backend.engine_gateway.application.events.training_events import (
    TrainingInitializedEvent,
    TrainingStartedEvent,
    TrainingPausedEvent,
    TrainingResumedEvent,
    TrainingStoppedEvent,
    TrainingCompletedEvent,
    TrainingFailedEvent,
)

from backend.training_service.application.event_handlers.base import (
    EngineEventHandler,
)

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)

from backend.training_service.domain.enums import (
    TrainingStatus,
)

from backend.training_service.domain.value_objects import (
    TrainingProgress,
)


class TrainingInitializedHandler(EngineEventHandler):

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ) -> None:

        self._training_repo = training_repo

    def handle(
        self,
        event: TrainingInitializedEvent,
    ) -> None:

        training = self._training_repo.get_by_id(
            event.run_id,
        )

        training.status = TrainingStatus.CREATED

        self._training_repo.update(
            training,
        )


class TrainingStartedHandler(EngineEventHandler):

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ) -> None:

        self._training_repo = training_repo

    def handle(
        self,
        event: TrainingStartedEvent,
    ) -> None:

        training = self._training_repo.get_by_id(
            event.run_id,
        )

        training.start()

        self._training_repo.update(
            training,
        )


class TrainingPausedHandler(EngineEventHandler):

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ) -> None:

        self._training_repo = training_repo

    def handle(
        self,
        event: TrainingPausedEvent,
    ) -> None:

        training = self._training_repo.get_by_id(
            event.run_id,
        )

        training.progress = TrainingProgress(
            step=event.global_step,
        )

        training.pause()

        self._training_repo.update(
            training,
        )


class TrainingResumedHandler(EngineEventHandler):

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ) -> None:

        self._training_repo = training_repo

    def handle(
        self,
        event: TrainingResumedEvent,
    ) -> None:

        training = self._training_repo.get_by_id(
            event.run_id,
        )

        training.progress = TrainingProgress(
            step=event.global_step,
        )

        training.resume()

        self._training_repo.update(
            training,
        )


class TrainingStoppedHandler(EngineEventHandler):

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ) -> None:

        self._training_repo = training_repo

    def handle(
        self,
        event: TrainingStoppedEvent,
    ) -> None:

        training = self._training_repo.get_by_id(
            event.run_id,
        )

        training.progress = TrainingProgress(
            step=event.global_step,
        )

        training.stop()

        self._training_repo.update(
            training,
        )


class TrainingCompletedHandler(EngineEventHandler):

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ) -> None:

        self._training_repo = training_repo

    def handle(
        self,
        event: TrainingCompletedEvent,
    ) -> None:

        training = self._training_repo.get_by_id(
            event.run_id,
        )

        training.progress = TrainingProgress(
            step=event.global_step,
        )

        training.status = TrainingStatus.COMPLETED
        training.finished_at = event.created_at

        self._training_repo.update(
            training,
        )


class TrainingFailedHandler(EngineEventHandler):

    def __init__(
        self,
        training_repo: TrainingRunRepository,
    ) -> None:

        self._training_repo = training_repo

    def handle(
        self,
        event: TrainingFailedEvent,
    ) -> None:

        training = self._training_repo.get_by_id(
            event.run_id,
        )

        training.progress = TrainingProgress(
            step=event.global_step,
        )

        training.fail()

        self._training_repo.update(
            training,
        )