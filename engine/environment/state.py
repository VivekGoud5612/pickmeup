from typing import Dict, List, Tuple, Any
from engine.agents.agent_data import AgentIdentity 
import numpy as np


GRID_DIM = 2
NUM_SKILLS = 3

class GameState:
    def __init__(self, num_agents : int, grid_size : int=10):
        self.grid_size = grid_size

        ## STATIC attributes
        self.roles = [''] * num_agents 
        self.teams = [''] * num_agents  # Assuming a specific team name signifies the agents team
        self.max_hp = np.zeros(num_agents, dtype = np.float32)
        self.max_stamina = np.zeros(num_agents, dtype = np.float32)
        self.defense = np.zeros(num_agents, dtype = np.float32)
        self.strength = np.zeros(num_agents, dtype = np.float32)


        ## run time mutable attributes
        self.positions = np.zeros((num_agents, GRID_DIM), dtype = np.int32)
        self.hp = np.zeros(num_agents, dtype = np.float32)
        self.stamina = np.zeros(num_agents, dtype = np.float32)
        self.cooldowns = np.zeros((num_agents, NUM_SKILLS), dtype = np.int32)
        self.is_blocking = np.zeros(num_agents, dtype = bool)
        
        self.team_visited_tiles: Dict[str, set] = {"Heroes": set(), "Boss": set()}  ## Unique tiles a team visited .. will visit after
        self.team_masks = {
            'Heroes' : np.zeros(num_agents, dtype = bool),   ## Fast check for team status of each agent.. Mostly unecessary
            'Monsters' : np.zeros(num_agents, dtype = bool)
        }

    def register_agents(self, agent_id:int, identity : AgentIdentity, team : str, start_position : Tuple[int, int]):
        
        self.roles[agent_id] = identity.role 
        self.teams[agent_id] = team
        self.max_hp[agent_id] = identity.stats.max_hp 
        self.max_stamina[agent_id] = identity.stats.attributes.stamina 
        self.defense[agent_id] = identity.stats.attributes.defense 
        self.strength[agent_id] = identity.stats.attributes.strength 

        self.positions[agent_id] = np.array(start_position)
        self.hpp[agent_id] = self.max_hp[agent_id] 
        self.stamina[agent_id] = self.max_stamina 
        
        self.team_masks[team][agent_id] = True 