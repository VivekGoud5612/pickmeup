from __future__ import annotations

from backend.training_service.application.handlers.base import EngineEventHandler 
from backend.engine_gateway.application.events.curriculum_events import (
    CurriculumAdvancedEvent,
)

class CurriculumAdvancedHandler(EngineEventHandler):

    def handle(
        self,
        event: CurriculumAdvancedEvent,
    ) -> None:
        ...