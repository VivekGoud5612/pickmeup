from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from backend.contracts.engine.enums import TrainingAlgorithm
from backend.training_service.domain.enums import TrainingStatus
from backend.contracts.engine.models.training_progress import TrainingProgress


@dataclass(slots=True, kw_only=True)
class TrainingRun:
    """
    Represents a single training session.
    """

    id: UUID = field(default_factory=uuid4)
    name: str
    algorithm: TrainingAlgorithm
    status: TrainingStatus = TrainingStatus.CREATED
    progress: TrainingProgress | None = None
    configuration_id: UUID | None = None
    latest_checkpoint_id: UUID | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    notes: str = ""

    def start(self) -> None:
        if self.status != TrainingStatus.CREATED:
            raise RuntimeError("Training has already been started")
        self.status = TrainingStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def pause(self) -> None:
        if self.status != TrainingStatus.RUNNING:
            raise RuntimeError("Training is not being run")
        self.status = TrainingStatus.PAUSED

    def resume(self) -> None:
        if self.status != TrainingStatus.PAUSED:
            raise RuntimeError("Training is not paused")
        self.status = TrainingStatus.RUNNING

    def stop(self) -> None:
        if self.status not in (TrainingStatus.RUNNING, TrainingStatus.PAUSED):
            raise RuntimeError("Training is not running or is not paused, so it cannot be stopped")
        self.status = TrainingStatus.STOPPED
        self.finished_at = datetime.now(UTC)

    def fail(self) -> None:
        self.status = TrainingStatus.FAILED
        self.finished_at = datetime.now(UTC)

    def advance_episode(self) -> None:
        if self.status != TrainingStatus.RUNNING:
            raise RuntimeError("Training is not running")
        if self.progress is None:
            raise RuntimeError("Training progress has not been initialized")
        self.progress = self.progress.advance_episode()

    def advance_step(self, amount: int = 1) -> None:
        if self.status != TrainingStatus.RUNNING:
            raise RuntimeError("training is not running")
        if self.progress is None:
            raise RuntimeError("Training progress has not been initialized")
        self.progress = self.progress.advance_step(amount)

    def attach_checkpoint(self, checkpoint_id: UUID) -> None:
        self.latest_checkpoint_id = checkpoint_id

    def attach_config(self, config_id: UUID) -> None:
        self.configuration_id = config_id
