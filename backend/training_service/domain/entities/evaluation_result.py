from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from training_service.domain.enums import EvaluationStatus
from training_service.domain.value_objects import PerformanceMetrics


@dataclass(slots=True, kw_only=True)
class EvaluationResult:
    """
    Represents the result of evaluating a training checkpoint.

    An evaluation is performed independently of training and
    summarizes the performance of a checkpoint.
    """

    checkpoint_id: UUID  ## So that we can differentiate between each checkpoint... and also because each checkpoint can have many evaluations..

    metrics: PerformanceMetrics   ## Note that these are mutable...

    num_episodes: int

    id: UUID = field(default_factory=uuid4)

    status: EvaluationStatus = EvaluationStatus.PENDING  ## By default.. will be pending till we start....

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    notes: str = ""

    def start(self) -> None:
        """
        Marks the evaluation as running.
        """

        if self.status != EvaluationStatus.PENDING:
            raise RuntimeError(
                "Evaluation has already started."
            )

        self.status = EvaluationStatus.RUNNING

    def complete(self) -> None:
        """
        Marks the evaluation as completed.
        """

        if self.status != EvaluationStatus.RUNNING:
            raise RuntimeError(
                "Evaluation is not running."
            )

        self.status = EvaluationStatus.COMPLETED

    def fail(self) -> None:
        """
        Marks the evaluation as failed.
        """

        self.status = EvaluationStatus.FAILED