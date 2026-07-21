"""
For objects inside the main DTOs
Like hyperparameters, reward weights etc
"""

from __future__ import annotations 

from dataclasses import dataclass 
from uuid import UUID 



@dataclass(slots = True, kw_only = True)
class HyperParametersRequest:
    """ 
    Important Hyperparameters
    """
    learning_rate : float  

    num_envs : int 
    
    gamma : float 
    gae_lambda : float 

    clip_range : float 
    entropy_coeff : float 

    batch_size : int 
    rollout_length : int  ## Because this is inherently different ffrom batch size...
    ppo_epochs : int  

    max_grad_norm : float 

    checkpoint_save_interval : int 


@dataclass(slots = True, kw_only = True)
class RewardWeightsRequest:
    """
    Reward weights to be used in request
    """

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


@dataclass(slots = True, kw_only = True)
class CurriculumSettingsRequest:
    """
    Cur settings to be used in request
    """

    boss_hp: float    

    spawn_radius : int  

    max_steps: int 

    difficulty_level: int  

    reward_scale: float = 1.0 

    grid_size : int

