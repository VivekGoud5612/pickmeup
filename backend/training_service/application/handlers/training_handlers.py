from __future__ import annotations

from backend.contracts.engine.events.training_events import (
    TrainingCompletedEvent,
    TrainingFailedEvent,
    TrainingInitializedEvent,
    TrainingPausedEvent,
    TrainingResumedEvent,
    TrainingStartedEvent,
    TrainingStoppedEvent,
)
from backend.training_service.application.handlers.base import EngineEventHandler


class TrainingInitializedHandler(EngineEventHandler):
    def handle(self, event: TrainingInitializedEvent) -> None:
        raise NotImplementedError


class TrainingStartedHandler(EngineEventHandler):
    def handle(self, event: TrainingStartedEvent) -> None:
        raise NotImplementedError


class TrainingPausedHandler(EngineEventHandler):
    def handle(self, event: TrainingPausedEvent) -> None:
        raise NotImplementedError


class TrainingResumedHandler(EngineEventHandler):
    def handle(self, event: TrainingResumedEvent) -> None:
        raise NotImplementedError


class TrainingStoppedHandler(EngineEventHandler):
    def handle(self, event: TrainingStoppedEvent) -> None:
        raise NotImplementedError


class TrainingCompletedHandler(EngineEventHandler):
    def handle(self, event: TrainingCompletedEvent) -> None:
        raise NotImplementedError


class TrainingFailedHandler(EngineEventHandler):
    def handle(self, event: TrainingFailedEvent) -> None:
        raise NotImplementedError
