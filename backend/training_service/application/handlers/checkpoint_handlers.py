from __future__ import annotations

from backend.contracts.engine.events.checkpoint_events import CheckpointCreatedEvent
from backend.training_service.application.handlers.base import EngineEventHandler


class CheckpointCreatedHandler(EngineEventHandler):
    def handle(self, event: CheckpointCreatedEvent) -> None:
        raise NotImplementedError
