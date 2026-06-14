from engine.actions.action_sequencer import ActionSequencer
from engine.agents.base_agent import BaseAgent 
from engine.environment.reward import RewardCalculator
from engine.environment.observation import ObservationBuilder
from typing import Dict, List, Any 
from engine.agents.agent_data import AgentRole, Teams, SkillTypes, AgentIdentity 
from engine.actions.action_handler import ActionTypes

import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
from enum import Enum


class ElementTypes(IntEnum):

    OBSERVATION = 0
    ROLES = 0
    REWARDS = 0
    DONES = 0
    ACTION_MASKS = 0
    ACTIVE_MASKS = 0

class AgentID(IntEnum):
    TANK = 0
    DEALER = 1
    HEALER = 2
    BOSS = 3

class Env(gym.Env):  ## A Multi Agent Gym Environment for 4 agent system
    ## This environment outputs local observations for the actor and global state for the global critic

    def __init__(self, grid_size : int = 20, max_steps : int = 200) ### We use the gym style APIs and returns .. here we have grid size as the parameter and max steps is the number of steps a certain game should run (That is the number of steps per epsode maybe)
        super().__init__()  ### Call the constructor of gym.Env

        self.grid_size = grid_size 
        self.max_steps = max_steps 
        self.step_count = 0
        self.num_agents = 4  ## For now let us define the number of agents in env 

        ## Instantiate all the data moving elements and some objects
        self.state = GameState(self.num_agents, self.grid_size)  ## Initialze game state..
        self.reward.calc = RewardCalculator()

        ## Spaces in Gym... need to see what they are.. They are something we use to like define what action and observation spaces are...
        self.action_spaces = spaces.MultiDiscrete([self.state.NUM_ACTIONS] * self.num_agents)  ## A space where there are 4 agents with 8 action options for each.. and MultiDiscrete maybe is something which is used to create a space object which is discrete across multiple dimensions..
        self.observation_space = spaces.Dict({
            "obs" : spaces.Box(low = -np.inf, high = np.inf, shape = (self.num_agents, ObservationBuilder.OBS_SIZE), dtype = np.float32).  ## Size (4, 24)
            "roles" : spaces.Box(low = 0, high = 3, shape = (self.num_agents, ObservationBuilder.NUM_ROLES), dtype = np.int32).
            "global_state" : spaces.Box(low = -np.inf, high = np.inf, shape = (self.num_agents, ObservationBuilder.GLOBAL_OBS_SIZE), dtype = np.float32)  ## Size (4, 96),
        })

        ### Create agent identities, these 4 agents run in all the environments .. and it doesn't matter if we define them in the reset or init as we store everything inside the shared memory and update rollout buffer from there.
        self.agent_dict = {
            AgentID.TANK : AgentRole.TANK,
            AgentID.DEALER : AgentRole.DEALER,
            AgentID.HEALER : AgentROle.HEALER,
            AgentID.BOSS : AgentRole.BOSS,
        }

    def reset(self, seed : Optional[int] = None, options : Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed = seed) ## Where is seed is used and what is seed I do not know.. Wrote in Notes, but is something to track the random resets and can be used to reproduce the same game..

        self.step_count = 0  # Update self.step count to 0 and start a new game
        self.state.reset()  ## Each agent start at some randomized position as described in state.reset()

        obs_dict = self._get_observations()   ### A observation dictionary we use for taking action in step...

        info = {
            "action_mask" : self._get_action_mask(),  ## We have multiple masks for each agent..
            "active_masks" : self._get_active_mask(),   ##There is only single active mask
            "handcrafted_potentials" : self._get_handcrafted_potential()  ## And we calculate handcrafted potentials here and the value comes in the main file..
        }

        return obs_dict, info
    
    def step(self, actions : np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, bool, Dict[str, np.ndarray]]:
        
        self.step_count += 1

        old_state_snapshot = stateops.state_snapshot(self.state)
        info["old_state" : old_state_snapshot]

        processed_mask = np.zeros(self.num_agents, dtype = bool)  ## Size of agents and actions is the same, as each agent takesup one aaction
        action_mask = self._get_action_mask()

        ActionSequencer.resolve_step(self.state, actions, processed_mask, action_mask)  ## Resolve step does everything.. inclusing action handling and everything...

