from __future__ import annotations

from training_service.application.handlers.base import EngineEventHandler 
from engine_gateway.application.events.curriculum_events import (
    CurriculumAdvancedEvent,
)

class CurriculumAdvancedHandler(EngineEventHandler):

    def handle(
        self,
        event: CurriculumAdvancedEvent,
    ) -> None:
        ...