from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from backend.training_service.domain.enums import (
    EvaluationStatus,
)

from backend.training_service.domain.value_objects import (
    PerformanceMetrics,
)


@dataclass(slots=True, kw_only=True)
class EvaluationResult:
    """
    Stores the result of evaluating a checkpoint.
    """

    checkpoint_id: UUID

    metrics: PerformanceMetrics

    id: UUID = field(default_factory=uuid4)

    status: EvaluationStatus = EvaluationStatus.COMPLETED

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    notes: str = ""

    def fail(self) -> None:
        self.status = EvaluationStatus.FAILED