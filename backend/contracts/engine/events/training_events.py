from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .base import EngineEvent

@dataclass(slots = True, frozen = True, kw_only = True)
class TrainingInitializedEvent(EngineEvent):
    run_id: UUID
   

@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingStartedEvent(EngineEvent):
    run_id: UUID


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingPausedEvent(EngineEvent):
    run_id: UUID
    global_step: int


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingResumedEvent(EngineEvent):
    run_id: UUID
    global_step: int


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingStoppedEvent(EngineEvent):
    run_id: UUID
    global_step: int


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingCompletedEvent(EngineEvent):
    run_id: UUID
    global_step: int
    total_episodes: int


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingFailedEvent(EngineEvent):
    run_id: UUID
    global_step: int
    reason: str
