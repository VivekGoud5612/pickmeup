from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from backend.engine_gateway.application.events.base import (
    EngineEvent,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class CurriculumAdvancedEvent(EngineEvent):
    """
    Published whenever the curriculum automatically
    advances to the next difficulty level.
    """

    run_id: UUID

    previous_level: int

    new_level: int

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )