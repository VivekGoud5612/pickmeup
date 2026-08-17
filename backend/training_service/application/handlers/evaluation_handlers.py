from __future__ import annotations

from backend.contracts.engine.events.evaluation_events import (
    EvaluationCompletedEvent,
    EvaluationStartedEvent,
)
from backend.training_service.application.handlers.base import EngineEventHandler


class EvaluationStartedHandler(EngineEventHandler):
    def handle(self, event: EvaluationStartedEvent) -> None:
        raise NotImplementedError


class EvaluationCompletedHandler(EngineEventHandler):
    def handle(self, event: EvaluationCompletedEvent) -> None:
        raise NotImplementedError
