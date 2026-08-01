from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from backend.engine_gateway.application.events.base import (
    EngineEvent,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingInitializedEvent(EngineEvent):
    """
    Published after the engine has been initialized and is
    ready to start training.
    """

    run_id: UUID

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingStartedEvent(EngineEvent):
    """
    Published when the training loop begins.
    """

    run_id: UUID

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingPausedEvent(EngineEvent):
    """
    Published when training is paused.
    """

    run_id: UUID

    global_step: int

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingResumedEvent(EngineEvent):
    """
    Published when training resumes.
    """

    run_id: UUID

    global_step: int

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingStoppedEvent(EngineEvent):
    """
    Published when training is stopped.
    """

    run_id: UUID

    global_step: int

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingCompletedEvent(EngineEvent):
    """
    Published when training completes successfully.
    """

    run_id: UUID

    global_step: int

    total_episodes: int

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class TrainingFailedEvent(EngineEvent):
    """
    Published when training terminates because of an error.
    """

    run_id: UUID

    global_step: int

    reason: str

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )