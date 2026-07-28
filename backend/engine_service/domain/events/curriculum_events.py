from __future__ import annotations

from dataclasses import dataclass

from .base import EngineEvent


@dataclass(slots=True, frozen=True)
class CurriculumAdvancedEvent(EngineEvent):
    previous_level: int
    
    new_level: int