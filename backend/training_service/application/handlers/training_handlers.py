from __future__ import annotations 

from engine_gateway.application.events.training_events import (
    TrainingInitializedEvent, TrainingStartedEvent, TrainingPausedEvent, TrainingResumedEvent, TrainingStoppedEvent, TrainingCompletedEvent, TrainingFailedEvent
)

from training_service.application.handlers.base import EngineEventHandler 


class TrainingInitializedHandler(EngineEventHandler):

    def handle(
        self,
        event: TrainingInitializedEvent,
    ) -> None:
        ...


class TrainingStartedHandler(EngineEventHandler):

    def handle(
        self,
        event: TrainingStartedEvent,
    ) -> None:
        ...


class TrainingPausedHandler(EngineEventHandler):

    def handle(
        self,
        event: TrainingPausedEvent,
    ) -> None:
        ...


class TrainingResumedHandler(EngineEventHandler):

    def handle(
        self,
        event: TrainingResumedEvent,
    ) -> None:
        ...


class TrainingStoppedHandler(EngineEventHandler):

    def handle(
        self,
        event: TrainingStoppedEvent,
    ) -> None:
        ...


class TrainingCompletedHandler(EngineEventHandler):

    def handle(
        self,
        event: TrainingCompletedEvent,
    ) -> None:
        ...


class TrainingFailedHandler(EngineEventHandler):

    def handle(
        self,
        event: TrainingFailedEvent,
    ) -> None:
        ...