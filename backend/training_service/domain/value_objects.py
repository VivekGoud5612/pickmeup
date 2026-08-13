"""
Value object representing the hyperparameters of a training configuration.

HyperParameters are immutable. They describe *how* a training run should
optimize the policy, but they do not have an identity or lifecycle.
"""

from dataclasses import dataclass 


@dataclass(slots = True, frozen = True, kw_only = True)
class HyperParameters:
    """
    MAPPO Hyperparameters
    """
    learning_rate : float

    num_envs : int 
    
    gamma : float 
    gae_lambda : float 

    clip_range : float 
    entropy_coeff : float 

    batch_size : int 
    rollout_length : int  ## Because this is inherently different from batch size...
    ppo_epochs : int  

    max_grad_norm : float 
    
    checkpoint_save_interval : int 


    def __post_init__(self) -> None:
        """
        We actually did that and created this.. not this function checks 
        or validates right after the class is initialized
        """
        self.validate()

    def validate(self) -> None:   ## We could also use __post_init__() to validate right after constructer. So it happens autoamticaly.. so let see if we dont have any uses of this vaidate or are forgeting to call that, then we can change to __pos_init__()
        """
        Validate if all hyperparameters are within their acceptable range
        and if not raise a value error
        """

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
            raise ValueError("Rollout length must be positive")

        if self.ppo_epochs <= 0:
            raise ValueError("PPO epochs must be positive.")

        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive.")




"""
Value object representing the reward shaping configuration
used by a training session.

RewardWeights are immutable and describe how different
environment events contribute to the total reward.
"""

@dataclass(slots = True, frozen = True, kw_only = True)
class RewardWeights:    

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

    def validate(self) -> None:
      """
        Validate reward coefficients. Which are needed to be strictly validated
        Other weights like dealer_damage or such could be negative as well, so we only validae needed information
        """

        if self.victory_bonus < 0:
            raise ValueError("Victory bonus cannot be negative.")

        if self.death_penalty > 0:
            raise ValueError("Death penalty must be non-positive.")

        if self.step_penalty > 0:
            raise ValueError("Step penalty must be non-positive.")

        if self.invalid_action_penalty > 0:
            raise ValueError("Invalid action penalty must be non-positive.")





"""
Value object representing the curriculum configuration used
during training.

Curriculum settings define the environment difficulty and are
immutable throughout a training configuration.
""" 

@dataclass(slots = True, frozen = True, kw_only = True)
class CurriculumSettings:
    """
    Environment curriculum configuration.
    """

    boss_hp: float    ## Later when my engine is complex with many things, like physics, obstacles, minions and such, this needs to be another object.. but for now boss hp is ok

    spawn_radius : int  ## How far should agents spawn...

    grid_size : int

    max_steps: int

    difficulty_level: int   ## Int for now but later we can define this as a enum if we have set levels of difficulty..

    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        Validate curriculum settings.
        """

        if self.boss_hp <= 0:
            raise ValueError("Boss HP must be positive.")

        if self.spawn_radius <= 0:
            raise ValueError("Spawn radius must be positive")

        if self.max_episode_steps <= 0:
            raise ValueError("Maximum episode steps must be positive.")

        if self.difficulty_level < 1:
            raise ValueError("Difficulty level must be at least 1.")

        if self.reward_scale <= 0:
            raise ValueError("Reward scale must be positive.")





from engine.utils.enums import AgentRole 


@dataclass(slots=True, kw_only=True)
class PerformanceMetrics:
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

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:

        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(
                "Win rate must lie in [0, 1]."
            )

        if not -1.0 <= self.explained_variance <= 1.0:
            raise ValueError(
                "Explained variance must lie in [-1, 1]."
            )

        if self.episode_length <= 0:
            raise ValueError(
                "Average episode length must be positive."
            )

    @property 
    def average_actor_loss(self) -> float:
        return sum(self.actor_losses.values()) / len(self.actor_losses)   ## So that storing becomes easy and we get the true average..

    @property 
    def average_critic_loss(self) -> float:
        return sum(self.critic_losses.values()) / len(self.critic_losses)