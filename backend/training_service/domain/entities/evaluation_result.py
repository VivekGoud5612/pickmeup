from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from backend.contracts.engine.enums import EvaluationStatus
from backend.contracts.engine.models.performance_metrics import PerformanceMetrics


@dataclass(slots=True, kw_only=True)
class EvaluationResult:
    checkpoint_id: UUID
    metrics: PerformanceMetrics | None = None
    num_episodes: int
    id: UUID = field(default_factory=uuid4)
    status: EvaluationStatus = EvaluationStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    notes: str = ""

    def start(self) -> None:
        if self.status != EvaluationStatus.PENDING:
            raise RuntimeError("Evaluation has already started.")
        self.status = EvaluationStatus.RUNNING

    def complete(self, metrics: PerformanceMetrics) -> None:
        if self.status != EvaluationStatus.RUNNING:
            raise RuntimeError("Evaluation is not running.")
        self.metrics = metrics
        self.status = EvaluationStatus.COMPLETED
        self.finished_at = datetime.now(UTC)

    def fail(self) -> None:
        self.status = EvaluationStatus.FAILED
        self.finished_at = datetime.now(UTC)
