import random
from typing import Dict, List, Tuple, Any
from engine.agents.agent_data import AgentIdentityFormat, AgentIdentity
from engine.utils.enums import AgentRole, Teams, SkillTypes, ActionTypes, AgentID
import numpy as np


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

SKILL_ACTION_MAP = {
    key : {v : k for k, v in ACTION_SKILL_MAP[key].items()} for key in ACTION_SKILL_MAP.keys()
}

class GameState:

    GRID_DIM = 2
    NUM_SKILLS = 3
    NUM_ACTIONS = 8
    NUM_TEAMS = 2
    NUM_ROLES = 4


    ACTION_TO_INDEX_MAP = {
        ActionTypes.BASIC : 0,
        ActionTypes.UTILITY : 1,
        ActionTypes.ULTIMATE : 2,
    }

    def __init__(self, num_agents : int, grid_size : int):
        self.grid_size = grid_size

        self.num_agents = num_agents
        ## STATIC attributes
        self.roles = np.zeros(GameState.NUM_ROLES, dtype = np.int32)  # There is no need for a string thing here because we can convert agentrole to IntEnum
        self.teams = np.zeros(num_agents, dtype = np.int32)  # Also converted to an intenum  # Assuming a specific team name signifies the agents team
        self.max_hp = np.zeros(num_agents, dtype = np.float32)
        self.max_stamina = np.zeros(num_agents, dtype = np.float32)
        self.defences = np.zeros(num_agents, dtype = np.float32)
        self.strengths = np.zeros(num_agents, dtype = np.float32)
        self.recovery_rates = np.zeros(num_agents, dtype = np.float32)

        ## Skill specific entries for use to use at runtime 
        self.skill_multiplier = np.zeros((num_agents, GameState.NUM_ACTIONS), dtype = np.float32)  ## This array has action spaces for total num agents, as skill multiplier is only for skills the first 5 are always 0.. but for easy tracking and retrieval and as every other action related arral also has 8 spaces.
        self.skill_min_ranges = np.zeros((num_agents, GameState.NUM_ACTIONS), dtype = np.float32)
        self.skill_max_ranges = np.zeros((num_agents, GameState.NUM_ACTIONS), dtype = np.float32)
        self.skill_stamina_cost = np.zeros((num_agents, GameState.NUM_ACTIONS), dtype = np.float32)
        self.team_masks = np.zeros((GameState.NUM_TEAMS, num_agents), dtype = bool)

        ## Load static entries
        self._load_static_elements() 

        ## run time mutable attributes
        self.positions = np.zeros((num_agents, GameState.GRID_DIM), dtype = np.int32)
        self.hp = np.zeros(num_agents, dtype = np.float32)
        self.stamina = np.zeros(num_agents, dtype = np.float32)
        self.cooldowns = np.zeros((num_agents, self.NUM_SKILLS), dtype = np.int32)
        self.is_blocking = np.zeros(num_agents, dtype = bool)
        self.is_invincible = np.zeros(num_agents, dtype = bool)
        self.invalid_actions = np.zeros(num_agents, dtype = bool)  # We use this in base agent, and there is no need for processed mask in environment, and we can simply assing invalid action penalty for invalid actions
        
        ##Event arrays for reward calculation
        self.damage_dealt = np.zeros(num_agents, dtype = np.float32)
        self.effective_heal = np.zeros(num_agents, dtype = np.float32)
        self.damage_reduction_by_block = np.zeros(num_agents, dtype = np.float32)
        self.damage_reduction_by_nullification = np.zeros(num_agents, dtype = np.float32)
        self.stamina_spent = np.zeros(num_agents, dtype = np.float32)
        self.damage_taken = np.zeros(num_agents, dtype = np.float32)
        self.exploration_bonus_triggered = np.zeros(num_agents, dtype = np.float32)

        self.team_visited_tiles = np.zeros((GameState.NUM_TEAMS, grid_size, grid_size), dtype = bool) # Where we have a grid of tiles for each team, and if they visited the that specific dim 1 and 2 for that team is True  # old {"Heroes": set(), "Boss": set()}  ## Unique tiles a team visited .. will visit after


    def _load_static_elements(self):

        for role, data in AgentIdentity.ROLES.items():  ## Instead of getting data during run time, we can simply get the whole data at the start of creation
        ## As role_id = agent_id for now.....
            self.roles[role] = role
            self.teams[role] = data['team']
            self.max_hp[role] = data['max_hp']
            self.max_stamina[role] = data['attributes'].stamina 
            self.defences[role] = data['attributes'].defence 
            self.strengths[role] = data['attributes'].strength 
            self.recovery_rates[role] = data['attributes'].recovery_rate 
            self.team_masks[data['team'], role] = 1.0

            for name, skill in data['skills'].items():  ## As we have 3 skills we need to loop over and add data

                action_idx = SKILL_ACTION_MAP[role][name]  ## We get the action ID corresponding to the skill name
                self.skill_multiplier[role, action_idx] = skill.strength_of_skill
                self.skill_min_ranges[role, action_idx] = skill.min_range 
                self.skill_max_ranges[role, action_idx] = skill.max_range 
                self.skill_stamina_cost[role, action_idx] = skill.stamina_cost 


    def reset(self, curriculum_level = 6):  ## This is the one called in reset and is in charge of resetting all the state elements which change during run time

        dynamic_boss_max_hp = min(1000.0, 300.0 + 100*(curriculum_level + 1))  ## Dynamically increase the max hp to 100 hp per curriculum level
        spawn_radius = min(10, 1 + curriculum_level)  ## Distance between 1-10 spawn pos

        self.max_hp[AgentID.BOSS] = dynamic_boss_max_hp  ## Update that specific boss hp so that we can safely adapt the boss hp based on curriculum level

        self.hp[:] = self.max_hp[:]  ## Copy as is, without making both the arrays point to the same memory
        self.stamina[:] = self.max_stamina[:]
        self.cooldowns[:].fill(0.0)  ## Let the cooldowns be anything , here we reset everything
        self.invalid_actions.fill(False)  ## This as well.... other things like is blocking, is invincible are taken care of in action sequencer, but we do that here as well.. no chances
        
        self.is_blocking.fill(False)  ## We set all the blockings to False at each new step, because the tank blocks once per step and has a cooldown. This needs to reset or else the tank stays on block the whole episode...
        self.is_invincible.fill(False)
        self.damage_dealt.fill(0.0)
        self.effective_heal.fill(0.0)
        self.damage_reduction_by_block.fill(0.0)
        self.damage_reduction_by_nullification.fill(0.0)
        self.stamina_spent.fill(0.0)
        self.damage_taken.fill(0.0)
        self.exploration_bonus_triggered.fill(0.0)  ## for agents which went to new state , give some sort of a bonus

        self.team_visited_tiles.fill(False)

        # 2. Vectorized Position Randomization
        occupied = set()   ## A different approach of the same randomized positions ... gemini gave this so decided to keep it...
        
        ## For now we spawn the agents near the boss...
        for a_idx in [3, 2, 1, 0]:
            if self.teams[a_idx] == Teams.MONSTERS:
                boss_pos = (np.random.randint(self.grid_size - 3, self.grid_size), np.random.randint(self.grid_size - 3, self.grid_size))
                self.positions[a_idx] = np.array(boss_pos)  ## Convert the tuple to array and also assign the boss pos as there is only one..
                occupied.add(boss_pos)
                self.team_visited_tiles[self.teams[a_idx], boss_pos[0], boss_pos[1]] = 1.0

            else:
                while True:
                    offset = np.random.randint(-spawn_radius, spawn_radius + 1, size = 2)  ## An offset of -2 to +2..random , for x and y Size 2 gives us 2 random offsets...
                    spawn_pos = np.clip(boss_pos + offset, 0, self.grid_size - 1) ## Add offset to boss_pos, and the value should be between 0 and grid_size - 1(19)..so just to safeguard pos
                    spawn_tuple = tuple(spawn_pos)  ## as it is an array from above

                    if spawn_tuple not in occupied:
                        self.positions[a_idx] = spawn_pos
                        occupied.add(spawn_tuple)

                        self.team_visited_tiles[self.teams[a_idx], spawn_tuple[0], spawn_tuple[1]] = 1.0
                        break

        '''for a_idx in range(self.num_agents):
            team = self.teams[a_idx]
            
            while True:
                if team == Teams.HEROES:
                    # Heroes spawn in the top 3 rows
                    pos = (np.random.randint(0, self.grid_size), np.random.randint(0, 3))
                else: 
                    # Boss spawns in the bottom right 3x3 corner
                    pos = (np.random.randint(self.grid_size - 3, self.grid_size), 
                           np.random.randint(self.grid_size - 3, self.grid_size))
                
                if pos not in occupied:
                    # Assign to array and mark occupied
                    self.positions[a_idx] = np.array(pos)
                    occupied.add(pos)
                    
                    # Mark the initial starting tile as visited for the team
                    self.team_visited_tiles[team, pos[0], pos[1]] = True
                    break'''

    def register_agent(self, agent_id : AgentID, identity : AgentIdentityFormat):

        self.roles[agent_id] = identity.role 
        self.teams[agent_id] = identity.team
        self.max_hp[agent_id] = identity.stats.max_hp 
        self.max_stamina[agent_id] = identity.stats.attributes.stamina 
        self.defences[agent_id] = identity.stats.attributes.defence 
        self.strengths[agent_id] = identity.stats.attributes.strength 

        ## For position randomizer
        if identity.team == Teams.HEROES:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, 2))   ## POsition randomizer... The very first step where position is taken on any row on first 3 cols..
            while pos in self.positions: ## If the position is already occupied
                pos = (random.randint(0, self.grid_size - 1), random.randint(0, 2))   ## If the randomized generated position is already occupied then we run the loop till we get a non occupied position
            self.positions[agent_id] = pos

        elif identity.team == Teams.MONSTERS:
            self.positions[agent_id] = (random.randint(self.grid_size-3, self.grid_size-1), random.randint(self.grid_size-3, self.grid_size-1)) ##  Assing a random position of boss in the last 3*3 grid of that big 20*20 grid

        self.positions[agent_id] = np.array(start_position)
        self.hp[agent_id] = self.max_hp[agent_id] 
        self.stamina[agent_id] = self.max_stamina[agent_id]
        
        self.team_masks[identity.team][agent_id] = True 
        self.team_visited_tiles[self.teams[agent_id]] = self.positions[agent_id]

        role = identity.role 
        skills = identity.stats.skills

        for action_enum, skill_enum in ACTION_SKILL_MAP[role].items() :  ## Where action enum is ActionTypes.ULTIMATE or some shit.. where skill name is the mapped skill name

            skill_data = skills[skill_enum]

            self.skill_multiplier[agent_id, action_enum] = skill_data.strength_of_skill 
            self.skill_min_ranges[agent_id, action_enum] = skill_data.min_range 
            self.skill_max_ranges[agent_id, action_enum] = skill_data.max_range 
            self.skill_stamina_cost[agent_id, action_enum] = skill_data.stamina_cost
