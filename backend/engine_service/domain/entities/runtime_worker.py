"""Execution-plane runtime identity.

This is intentionally not a TrainingRun.  It references a control-plane run
and owns only the lifecycle of its execution resource.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from backend.contracts.engine.enums import EngineStatus
from backend.contracts.engine.models.checkpoint_metadata import CheckpointMetadata
from backend.contracts.engine.models.training_progress import TrainingProgress
from backend.engine_service.domain.execution_handle import ExecutionHandle


@dataclass(slots=True, kw_only=True)
class RuntimeWorker:
    run_id: UUID
    execution_handle: ExecutionHandle
    status: EngineStatus = EngineStatus.CREATED
    progress: TrainingProgress = field(
        default_factory=lambda: TrainingProgress(episode=0, step=0)
    )
    latest_checkpoint: CheckpointMetadata | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def start(self) -> None:
        if self.status not in (EngineStatus.CREATED, EngineStatus.INITIALIZED):
            raise RuntimeError("Runtime worker has already been started")
        self.execution_handle.start()
        self.status = EngineStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def pause(self) -> None:
        if self.status != EngineStatus.RUNNING:
            raise RuntimeError("Runtime worker is not running")
        self.execution_handle.pause()
        self.status = EngineStatus.PAUSED

    def resume(self) -> None:
        if self.status != EngineStatus.PAUSED:
            raise RuntimeError("Runtime worker is not paused")
        self.execution_handle.resume()
        self.status = EngineStatus.RUNNING

    def stop(self) -> None:
        if self.status not in (EngineStatus.RUNNING, EngineStatus.PAUSED):
            raise RuntimeError("Runtime worker is not active")
        self.execution_handle.stop()
        self.status = EngineStatus.STOPPED
        self.finished_at = datetime.now(UTC)
