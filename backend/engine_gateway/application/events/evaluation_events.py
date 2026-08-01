from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from backend.engine_gateway.application.events.base import (
    EngineEvent,
)

from backend.engine_gateway.application.dto.nested import (
    PerformanceMetrics,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class EvaluationStartedEvent(EngineEvent):
    """
    Published when evaluation of a checkpoint begins.
    """

    run_id: UUID

    checkpoint_path: Path

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class EvaluationCompletedEvent(EngineEvent):
    """
    Published when checkpoint evaluation finishes.
    """

    run_id: UUID

    checkpoint_path: Path

    metrics: PerformanceMetrics

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )