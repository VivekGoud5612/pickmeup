from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import EngineEvent
from engine.utils.enums import AgentRole


@dataclass(slots=True, frozen=True, kw_only = True)
class EvaluationStartedEvent(EngineEvent):
    checkpoint_path: Path


@dataclass(slots=True, frozen=True, kw_only = True)
class EvaluationCompletedEvent(EngineEvent):
    checkpoint_path: Path
    average_reward: float
    win_rate: float
    episode_length: float

    actor_losses: dict[AgentRole, float]  
    critic_losses: dict[AgentRole, float]  

    entropy: float
    explained_variance: float