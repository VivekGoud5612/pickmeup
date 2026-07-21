from __future__ import annotations 

from backend.engine_gateway.application.events.base import EngineEvent 

@dataclass(slots = True, frozen = True, kw_only = True)
class TrainingInitializedEvent(EngineEvent):
    run_name : str 
   

@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingStartedEvent(EngineEvent):
    run_name: str


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingPausedEvent(EngineEvent):
    global_step: int


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingResumedEvent(EngineEvent):
    global_step: int


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingStoppedEvent(EngineEvent):
    global_step: int


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingCompletedEvent(EngineEvent):
    global_step: int
    total_episodes: int


@dataclass(slots=True, frozen=True, kw_only = True)
class TrainingFailedEvent(EngineEvent):
    global_step: int
    reason: str