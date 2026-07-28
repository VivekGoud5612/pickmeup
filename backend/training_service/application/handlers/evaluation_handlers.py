from __future__ import annotations

from engine_gateway.application.events.evaluation_events import (
    EvaluationStartedEvent, EvaluationCompletedEvent,
)
from training_service.application.handlers.base import EngineEventHandler 


class EvaluationStartedHandler(EngineEventHandler):

    def handle(
        self,
        event: EvaluationStartedEvent,
    ) -> None:
        ...


class EvaluationCompletedHandler(EngineEventHandler):

    def handle(
        self,
        event: EvaluationCompletedEvent,
    ) -> None:
        ...