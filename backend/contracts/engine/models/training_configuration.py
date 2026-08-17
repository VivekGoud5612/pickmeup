from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from backend.contracts.engine.enums import TrainingAlgorithm


@dataclass(slots=True, frozen=True, kw_only=True)
class HyperParameters:
    learning_rate: float
    num_envs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coeff: float
    batch_size: int
    rollout_length: int
    ppo_epochs: int
    max_grad_norm: float
    checkpoint_save_interval: int

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("Gamma must lie in (0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("GAE Lambda must lie in [0, 1].")
        if self.clip_range <= 0:
            raise ValueError("Clip range must be positive.")
        if self.entropy_coeff < 0:
            raise ValueError("Entropy coefficient cannot be negative.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.rollout_length <= 0:
            raise ValueError("Rollout length must be positive.")
        if self.ppo_epochs <= 0:
            raise ValueError("PPO epochs must be positive.")
        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive.")


@dataclass(slots=True, frozen=True, kw_only=True)
class RewardWeights:
    dealer_damage: float
    tank_damage: float
    healer_healing: float
    boss_damage: float
    tank_block: float
    healer_damage: float
    boss_healing: float
    death_penalty: float
    step_penalty: float
    victory_bonus: float
    invalid_action_penalty: float

    def __post_init__(self) -> None:
        if self.victory_bonus < 0:
            raise ValueError("Victory bonus cannot be negative.")
        if self.death_penalty > 0:
            raise ValueError("Death penalty must be non-positive.")
        if self.step_penalty > 0:
            raise ValueError("Step penalty must be non-positive.")
        if self.invalid_action_penalty > 0:
            raise ValueError("Invalid action penalty must be non-positive.")


@dataclass(slots=True, frozen=True, kw_only=True)
class CurriculumSettings:
    boss_hp: float
    spawn_radius: int
    grid_size: int
    max_steps: int
    difficulty_level: int
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.boss_hp <= 0:
            raise ValueError("Boss HP must be positive.")
        if self.spawn_radius <= 0:
            raise ValueError("Spawn radius must be positive.")
        if self.max_steps <= 0:
            raise ValueError("Maximum episode steps must be positive.")
        if self.difficulty_level < 1:
            raise ValueError("Difficulty level must be at least 1.")
        if self.reward_scale <= 0:
            raise ValueError("Reward scale must be positive.")


@dataclass(slots=True, kw_only=True)
class TrainingConfiguration:
    id: UUID = field(default_factory=uuid4)
    family_id: UUID = field(default_factory=uuid4)
    is_training: bool = False
    version: int = 1
    name: str
    algorithm: TrainingAlgorithm
    hyperparameters: HyperParameters
    reward_weights: RewardWeights
    curriculum: CurriculumSettings
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
