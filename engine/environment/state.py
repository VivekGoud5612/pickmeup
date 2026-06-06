from typing import Dict, List, Tuple, Any
from engine.agents.agent_data import AgentIdentityFormat, AgentRole, Teams
import numpy as np
from engine.actions.action_sequencer import ActionSequencer 


GRID_DIM = 2
NUM_SKILLS = 3
NUM_ACTIONS = 8
NUM_TEAMS = 2

class GameState:

    ACTION_SKILL_MAP = {
        AgentRole.TANK : {
            ActionTypes.BASIC : SkillTypes.BASIC,
            ActionTypes.UTILITY : SkillTypes.BLOCK,
            ActionTypes.ULTIMATE : SkillTypes.INVINCIBLE,
        },

        AgentRole.DEALER : {
            ActionTypes.BASIC : SkillTypes.BASIC,
            ActionTypes.UTILITY : SkillTypes.PIERCE,
            ActionTypes.ULTIMATE : SkillTypes.SPECIAL,
        },

        AgentRole.HEALER : {
            ActionTypes.BASIC : SkillTypes.BASIC,
            ActionTypes.UTILITY : SkillTypes.HEAL,
            ActionTypes.ULTIMATE : SkillTypes.ALL_HEAL,
        },

        AgentRole.BOSS : {
            ActionTypes.BASIC : SkillTypes.BASIC,
            ActionTypes.UTILITY : SkillTypes.REGENERATE,
            ActionTypes.ULTIMATE : SkillTypes.AOE,
        },
    }

    def __init__(self, num_agents : int, grid_size : int=10):
        self.grid_size = grid_size

        ## STATIC attributes
        self.roles = np.zeros(num_agents, dtype = np.int32)  # There is no need for a string thing here because we can convert agentrole to IntEnum
        self.teams = np.zeros(num_agents, dtype = np.int32)  # Also converted to an intenum  # Assuming a specific team name signifies the agents team
        self.max_hp = np.zeros(num_agents, dtype = np.float32)
        self.max_stamina = np.zeros(num_agents, dtype = np.float32)
        self.defences = np.zeros(num_agents, dtype = np.float32)
        self.strengths = np.zeros(num_agents, dtype = np.float32)
        self.recovery_rates = np.zeros(num_agents, dtype = np.float32)

        ## Skill specific entries for use to use at runtime 
        self.skill_multiplier = np.zeros((num_agents, NUM_ACTIONS), dtype = np.float32)  ## This array has action spaces for total num agents, as skill multiplier is only for skills the first 5 are always 0.. but for easy tracking and retrieval and as every other action related arral also has 8 spaces.
        self.skill_min_ranges = np.zeros((num_agents, NUM_ACTIONS), dtype = np.float32)
        self.skill_max_ranges = np.zeros((num_agents, NUM_ACTIONS), dtype = np.float32)
        self.skill_stamina_cost = np.zeros((num_agents, NUM_ACTIONS), dtype = np.float32)

        ## run time mutable attributes
        self.positions = np.zeros((num_agents, GRID_DIM), dtype = np.int32)
        self.hp = np.zeros(num_agents, dtype = np.float32)
        self.stamina = np.zeros(num_agents, dtype = np.float32)
        self.cooldowns = np.zeros((num_agents, NUM_SKILLS), dtype = np.int32)
        self.is_blocking = np.zeros(num_agents, dtype = bool)
        self.is_invincible = np.zeros(num_agents, dtype = bool)
        
        ##Evnet arrays for reward calculation
        self.damage_dealt = np.zeros(num_agents, dtype = np.float32)
        self.effective_healing = np.zeros(num_agents, dtype = np.float32)
        self.damage_reduction_by_block = np.zeros(num_agents, dtype = np.float32)
        self.damage_reduction_by_nullification = np.zeros(num_agents, dtype = np.float32)
        self.stamina_spent = np.zeros(num_agents, dtype = np.float32)
        self.damage_taken = np.zeros(num_agents, dtype = np.float32)

        self.team_visited_tiles = np.zeros((NUM_TEAMS, grid_size, grid_size), dtype = bool) # Where we have a grid of tiles for each team, and if they visited the that specific dim 1 and 2 for that team is True  # old {"Heroes": set(), "Boss": set()}  ## Unique tiles a team visited .. will visit after
        self.team_masks = np.zeros((NUM_TEAMS, num_agents), dtype = bool)

    def register_agent(self, agent_id:int, identity : AgentIdentityFormat, team : Teams, start_position : Tuple[int, int]):
        
        self.roles[agent_id] = identity.role 
        self.teams[agent_id] = team
        self.max_hp[agent_id] = identity.stats.max_hp 
        self.max_stamina[agent_id] = identity.stats.attributes.stamina 
        self.defences[agent_id] = identity.stats.attributes.defence 
        self.strengths[agent_id] = identity.stats.attributes.strength 

        self.positions[agent_id] = np.array(start_position)
        self.hp[agent_id] = self.max_hp[agent_id] 
        self.stamina[agent_id] = self.max_stamina[agent_id]
        
        self.team_masks[team][agent_id] = True 

        role = identity.role 
        skills = identity.stats.skills

        for action_enum, skill_enum in ACTION_SKILL_MAP[role].items() :  ## Where action enum is ActionTypes.ULTIMATE or some shit.. where skill name is the mapped skill name

            action_idxg = action_enum.value   # Action enums value is just a string
            skill_data = skills[skill_enum]

            self.skill_multipler[agent_id, action_idx] = skill_data.strength_of_skill 
            self.skill_min_range[agent_id, action_idx] = skill_data.min_range 
            self.skill_max_range[agent_id, action_idx] = skill_data.max_range 
            self.skill_stamina_cost[agent_id, action_idx] = skill_data.stamina_cost
