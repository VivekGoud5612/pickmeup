from __future__ import annotations
from dataclasses import dataclass 
from uuid import UUID 


@dataclass(slots=True, kw_only=True)
class PerformanceMetricsRequest:
    """
    Snapshot of the current training performance.

    This value object represents learning metrics collected
    during training. It contains no identity and is immutable.
    """

    average_reward: float

    actor_losses: dict[AgentRole, float]  ## Scalable with any numbe of actor critic losses...
    critic_losses: dict[AgentRole, float]  ## As we have our actors and critics for each role...

    entropy: float
    explained_variance: float

    win_rate: float

    episode_length: float  ## All of these are either averaged over 100 games or such, because there is no meaning storng them per step..