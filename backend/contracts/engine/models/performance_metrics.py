from __future__ import annotations

from dataclasses import dataclass
from backend.contracts.engine.enums import AgentRole


@dataclass(slots=True, frozen=True, kw_only=True)
class PerformanceMetrics:
    average_reward: float
    actor_losses: dict[AgentRole, float]
    critic_losses: dict[AgentRole, float]
    entropy: float
    explained_variance: float
    win_rate: float
    episode_length: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError("Win rate must lie in [0, 1].")
        if not -1.0 <= self.explained_variance <= 1.0:
            raise ValueError("Explained variance must lie in [-1, 1].")
        if self.episode_length <= 0:
            raise ValueError("Average episode length must be positive.")

    @property
    def average_actor_loss(self) -> float:
        return sum(self.actor_losses.values()) / len(self.actor_losses) if self.actor_losses else 0.0

    @property
    def average_critic_loss(self) -> float:
        return sum(self.critic_losses.values()) / len(self.critic_losses) if self.critic_losses else 0.0
