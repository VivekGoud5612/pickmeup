from engine.actions.action_sequencer import ActionSequencer
from engine.environment.reward import RewardCalculator
from engine.environment.observation import ObservationBuilder
from typing import Dict, List, Any, Optional, Tuple
from engine.utils.enums import AgentRole, Teams, SkillTypes , ActionTypes, AgentID
from engine.agents.agent_data import AgentIdentity
from engine.environment.state import GameState
from engine.environment.state_ops import StateOperations as stateops

import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
from enum import Enum




class Env(gym.Env):  ## A Multi Agent Gym Environment for 4 agent system
    ## This environment outputs local observations for the actor and global state for the global critic

    def __init__(self, grid_size : int = 20, max_steps : int = 200): ### We use the gym style APIs and returns .. here we have grid size as the parameter and max steps is the number of steps a certain game should run (That is the number of steps per epsode maybe)
        super().__init__()  ### Call the constructor of gym.Env

        self.grid_size = grid_size 
        self.max_steps = max_steps 
        self.step_count = 0
        self.num_agents = 4  ## For now let us define the number of agents in env 

        ## Instantiate all the data moving elements and some objects
        self.state = GameState(self.num_agents, self.grid_size)  ## Initialze game state..
        self.reward_calc = RewardCalculator()

        ## Spaces in Gym... need to see what they are.. They are something we use to like define what action and observation spaces are...
        self.action_spaces = spaces.MultiDiscrete([self.state.NUM_ACTIONS] * self.num_agents)  ## A space where there are 4 agents with 8 action options for each.. and MultiDiscrete maybe is something which is used to create a space object which is discrete across multiple dimensions..
        self.observation_space = spaces.Dict({
            "obs" : spaces.Box(low = -np.inf, high = np.inf, shape = (self.num_agents, ObservationBuilder.OBS_SIZE), dtype = np.float32),  ## Size (4, 24)
            "roles" : spaces.Box(low = 0, high = 3, shape = (self.num_agents, self.state.NUM_ROLES), dtype = np.int32),
            "global_state" : spaces.Box(low = -np.inf, high = np.inf, shape = (self.num_agents, ObservationBuilder.GLOBAL_STATE_SIZE), dtype = np.float32),  ## Size (4, 96),
        })  ## Again note that we only one single critic, and the values and such are repeated for batch in rollout buffer...

        ### Create agent identities, these 4 agents run in all the environments .. and it doesn't matter if we define them in the reset or init as we store everything inside the shared memory and update rollout buffer from there.
        self.agent_dict = {
            AgentID.TANK : AgentRole.TANK,
            AgentID.DEALER : AgentRole.DEALER,
            AgentID.HEALER : AgentRole.HEALER,
            AgentID.BOSS : AgentRole.BOSS,
        }

        self.episode_rewards = np.zeros(self.num_agents, dtype = np.float32)  ## To track the episodic reward in main, we calculate per env rewards ehre and store it in info

    def reset(self, curriculum_level : int = 10, seed : Optional[int] = None, options : Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed = seed) ## Where is seed is used and what is seed I do not know.. Wrote in Notes, but is something to track the random resets and can be used to reproduce the same game..

        self.step_count = 0  # Update self.step count to 0 and start a new game
        self.state.reset(curriculum_level)  ## Each agent start at some randomized position as described in state.reset()

        obs_dict = self._get_observations()   ### A observation dictionary we use for taking action in step...

        self.episode_rewards = np.zeros(self.num_agents, dtype = np.float32)  ## To track the episodic reward in main, we calculate per env rewards ehre and store it in info
        
        info = {
            "action_masks" : self._get_action_masks(),  ## We have multiple masks for each agent..
            "active_masks" : self._get_active_masks(),   ##There is only single active mask  ## And we calculate handcrafted potentials here and the value comes in the main file..
        }

        return obs_dict, info
    
    def step(self, actions : np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, bool, Dict[str, np.ndarray]]:
        
        self.step_count += 1
        info = {}

        old_state_snapshot = stateops.state_snapshot(self.state)

        processed_mask = np.zeros(self.num_agents, dtype = bool)  ## Size of agents and actions is the same, as each agent takesup one aaction
        action_masks = self._get_action_masks()
        active_masks = self._get_active_masks()

        ActionSequencer.resolve_step(self.state, actions, processed_mask, action_masks)  ## Resolve step does everything.. inclusing action handling and everything...

        reward_dict = self.reward_calc.calculate_decomposed_reward(old_state_snapshot, self.state, 0) ## Phase 0 here... The best thing instead of writing a new function, we can keep on running this for the first phase... But if we are going to use 

        rewards = reward_dict['combat_rewards'] + reward_dict['terminal_rewards']
        self.episode_rewards += rewards 

        is_terminal = stateops.is_terminal(self.state)   ## IF the match ended... where all the members of one team lost..
        truncated = bool(self.step_count >= self.max_steps)  ## If the current step is equal to greater than the current step count then we will stop the game..

        terminated_array = np.zeros(self.num_agents, dtype = np.float32)  ## An array to check for dead agents in this round and if terminal, then terminate all the agents (For active mask)
        for a_idx in range(self.num_agents):   
            was_alive = old_state_snapshot.hp[a_idx] > 0
            is_alive = self.state.hp[a_idx] > 0

            if (was_alive and not is_alive) or is_terminal or truncated:  ## THe reason for adding truncated was to consider this for next dones as well, when then the game is truncated we need to have the dones to be True for the next dones in GAE computation..
                terminated_array[a_idx] = 1.0 

        next_obs_dict = self._get_observations()  ## fOr the next set of actions to come, we need to send these out and give them to our actor

        info = {
            "action_masks" : action_masks,
            "active_masks" : active_masks,
            "old_hand" : reward_dict['old_handcrafted'],  ## Store the old and new potentials directly, No need for all of that reward dict and such.. This is easy to convert to numpy and share across memory
            "new_hand" : reward_dict['new_handcrafted'],
            "terminal" : is_terminal,
        }


        if is_terminal or truncated:
            info['episode_rewards'] = self.episode_rewards.copy()

            team = stateops.get_winning_team(self.state)
            
            if team == Teams.HEROES:
                info['win_rate'] = 1.0
            
            elif team == Teams.MONSTERS:
                info['win_rate'] = -1.0

            else:
                info['win_rate'] = 0.0

            self.episode_rewards = np.zeros(self.num_agents, dtype = np.float32)  ## We reset so that the next game

            info['episode_length'] = self.step_count
            info['boss_hp'] = self.state.hp[AgentID.BOSS]

        return next_obs_dict, rewards, terminated_array, truncated, info ## Classic gym style returns... No need for is_terminal...


    def close(self):   ## Simple function which does nothing, but closes the current environment (Used after all the work is complete and it is time to close the game..)
        pass 

    
    def _get_observations(self) -> Dict[str, np.ndarray]:

        env_obs = np.zeros((self.num_agents, ObservationBuilder.OBS_SIZE), dtype = np.float32)  ## Contains all the individual observations. so the size of this array after transformation is (4, 24).. for actors
        env_roles = np.zeros((self.num_agents, self.state.NUM_ROLES), dtype = np.int32)
        global_state_obs = np.zeros(ObservationBuilder.GLOBAL_STATE_SIZE, dtype = np.float32)  ## The same size, contains (96,) size of information

        for a_idx in self.agent_dict.keys():
            obs, role_ids = ObservationBuilder.build_partial_obs(self.state, a_idx)
            env_obs[a_idx] = obs 
            env_roles[a_idx] = role_ids 

        global_state_obs = ObservationBuilder.build_full_state(self.state)  ## Building the full observation for the whole state at once for critic
        global_state = np.tile(global_state_obs, (self.num_agents, 1))  ## TIle global state to each agent, so that we get a state array of size (num_agents, 96)

        return {
            "obs" : env_obs,
            "role_ids" : env_roles,
            "global_state_obs" : global_state, 
        }
        
    def _get_action_masks(self):
        action_masks = np.zeros((self.num_agents, self.state.NUM_ACTIONS), dtype = np.bool_)  ## Contains action mask for each agent

        for a_idx in self.agent_dict.keys():
            action_masks[a_idx] = stateops.get_action_mask(self.state, a_idx)
        return action_masks 

    def _get_active_masks(self):
        active_masks = np.zeros(self.num_agents, dtype = bool)
        active_masks = stateops.get_alive_mask(self.state)

        return active_masks 

    


