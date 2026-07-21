from __future__ import annotations 

from training_service.application.handlers.base import EngineEventHandler 
from engine_gateway.application.events.checkpoint_events import (
    CheckpointCreatedEvent,
)


class CheckpointCreatedHandler(EngineEventHandler):

    def handle(
        self,
        event: CheckpointCreatedEvent,
    ) -> None:
        ...