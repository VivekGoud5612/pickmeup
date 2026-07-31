

from dataclasses import dataclass


@dataclass(slots = True, frozen = True, kw_only = True)
class Hyperparameters:
    """MAPPO Hyperparameters"""

    learning_rate : float

    num_envs : int

    gamma : float
    gae_lambda : float

    clip_range : float
    entropy_coeff : float

    batch_size : int
    rollout_length : int
    ppo_epochs : int

    max_grad_norms : float

    checkpoint_save_interval : int

    def __post_init__(self) -> None:
        """Validate the fields after initializing, for storing and using only clean data"""
        self.validate()

    def validate(self) -> None:
        """Validate if all hyperparameters are in acceptable range"""

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("Gamma must lie in (0, 1]")

        if not 0.0 <=self.gae_lambda <= 1.0:
            raise ValueError("Gae Lambda must lie in [0, 1]")
        
        if self.clip_range <= 0.0:
            raise ValueError("Clip range must be positive")

        if self.entropy_coeff < 0.0:
            raise ValueError("Entropy Coefficient must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch Size must be positive")
        
        if self.rollout_length <= 0:
            raise ValueError("Rollout Length must be positive")

        if self.ppo_epochs <= 0:
            raise ValueError("PPO epochs must be positive")
        
        if self.max_grad_norms <= 0:
            raise ValueError("Max gradient norm must be positive")


@dataclass(slots = True, frozen = True, kw_only = True)
class RewardWeights:
    """Represents the reward weights for each type of action, and result of that of action"""

    dealer_damage : float
    tank_damage : float
    healer_healing : float
    boss_damage : float

    tank_block : float
    healer_damage : float
    boss_healing : float

    death_penalty : float
    step_penalty : float

    victory_bonus : float

    invalid_action_penalty : float

    def __post_init__(self) -> None:
        self.validate()

    def validate() -> None:

        if self.death_penalty > 0.0:
            raise ValueError("Death penalty cannot be positive, must be negative")
        
        if self.step_penalty > 0.0:
            raise ValueError("Step penalty must be negative")

        if self.victory_bonus < 0.0:
            raise ValueError("Victory bonus cannot be negative")

        if self.invalid_action_penalty > 0.0:
            raise ValueError("Invalid Action Penalty must be negative")


@dataclass(slots = True, frozen = True, kw_only = True)
class CuriculumSettings:
    """Represents the curiculum settings of the environment"""

    boss_hp : float

    spawn_radius : float

    grid_size : int

    max_episode_steps : int

    difficulty_level : int

    reward_scale : float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        
        if self.boss_hp <= 0:
            raise ValueError("Boss HP must be positive")

        if self.spawn_radius <= 0.0:
            raise ValueError("Spawn radius must be positive")

        if self.grid_size <= 0:
            raise ValueError("Grid size must be positive")
        
        if self.max_episode_steps <= 0:
            raise ValueError("Max Episode Steps must be positive")
        
        if self.difficulty_level < 1:
            raise ValueError("Difficulty level must be atleast 1")
        
        if self.reward_scale < 0:
            raise ValueError("Reward Scale must be positive")


@dataclass(slots = True, kw_only = True)
class TrainingProgress:
    "Represents the state of the training at that time"

    episode : int

    step : int

    total_episodes : int | None = None

    total_steps : int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:

        if self.episode < 0:
            raise ValueError("Current episode cannot be negative")
        
        if self.step < 0:
            raise ValueError("Current step cannot be negative")

        if self.total_episodes is not None and self.total_episodes < 0 :
            raise ValueError("Total episodes must be positive")

        if self.total_steps is not None and self.total_steps < 0:
            raise ValueError("Total Steps must be positive")

    @property
    def progress_percentage(self) -> float | None:

        if self.total_episodes is None:
            return None

        return (self.episode / self.total_episodes) * 100.0

    def advance_step(self) -> "TrainingProgress"





