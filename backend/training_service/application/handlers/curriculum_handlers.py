from __future__ import annotations

from backend.contracts.engine.events.curriculum_events import CurriculumAdvancedEvent
from backend.training_service.application.handlers.base import EngineEventHandler


class CurriculumAdvancedHandler(EngineEventHandler):
    def handle(self, event: CurriculumAdvancedEvent) -> None:
        raise NotImplementedError
