from __future__ import annotations 

from backend.training_service.application.handlers.base import EngineEventHandler 
from backend.engine_gateway.application.events.checkpoint_events import (
    CheckpointCreatedEvent,
)


class CheckpointCreatedHandler(EngineEventHandler):

    def handle(
        self,
        event: CheckpointCreatedEvent,
    ) -> None:
        ...