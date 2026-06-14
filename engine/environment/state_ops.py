import numpy as np 
from engine.environment.state import GameState
from typing import Optional
import copy

class StateOperations:

    ### A stateless object with all the helper functions we use with State. GameState is just a data vault and these are the functions that go along with those array data.


    ### BASIC QUERIES
    @staticmethod
    def is_alive(state : GameState, agent_id : int) -> bool :
        return state.hp[agent_id] > 0  ## Note that state.hp is a numpy array here. 

    @staticmethod 
    def get_alive_mask(state : GameState) :
            
            alive_mask = state.hp > 0    ## Gives the array with all the indexes of agents, and if alive we get True else false
            return alive_mask

    @staticmethod 
    def get_alive_agent_ids(state : GameState):
        alive_mask = StateOperations.get_alive_mask(state)
        
        return np.where(alive_mask)[0]

    @staticmethod 
    def get_alive_team_agent_ids(state : GameState, team : Teams):

        alive_mask = state.hp > 0   ## Including boss 

        team_agents = state.team_masks[team]    # only the agents in team are True, others are False

        return np.where(team_agents & alive_mask)[0]

    @staticmethod 
    def get_team_alive_mask(state : GameState, team : Teams):
        alive_mask = state.hp > 0 

        team_agents = state.team_masks[team] 
        valid_mask = alive_mask & team_agents    ### Teke only the agents in that team.. so we only take the values of the mask which are true.

        return valid_mask

    @staticmethod 
    def get_alive_allies(state : GameState, agent_id : int) -> np.ndarray:

        self_team = state.teams[agent_id]  ## self_team is a string with team name
        team_mask = state.team_masks[self_team]  # We get the team mask from the above team.

        alive_mask = state.hp > 0 # Same as above 

        valid_mask = team_mask & alive_mask 
        valid_mask[agent_id] = False 

        return np.where(valid_mask)[0]

    @staticmethod 
    def get_agents_other_than_self(state : GameState, agent_id : int):
        self_mask = np.zeros(state.num_agents, dtype = np.bool_)
        self_mask[agent_id] = True 

        not_self_mask = ~self_mask 

        return list(np.where(not_self_mask)[0])

    @staticmethod 
    def get_alive_enemies(state : GameState, agent_id : int) -> np.ndarray:

        self_team = state.teams[agent_id]

        enemy_mask = ~state.team_masks[self_team]  ## For opponent team we simply flip the bits
        alive_mask = state.hp > 0 

        return np.where(enemy_mask & alive_mask)[0]

    @staticmethod 
    def get_occupied_positions(state : GameState, exclude_id : int = -1) -> set:  # Exclude Id is helpful if we want to exclude some agent in their position calculation

        active_mask = state.hp > 0
        if exclude_id >= 0:
            active_mask[exclude_id] = False  ## Excluding that particular agent with ID exlcude ID.
        
        occupied_array = state.positions[active_mask] ## Called boolean indexing... We get the positions of only those that are alive (or are True in alive_mask)..

        return set(map(tuple, occupied_array))
        ## Map - That is iterate over each inernal array of occupied array that is for each position [x,y] we change it to tuple - (x,y)
        ## Then we wrap it in a set object, because lookup in a set takes O(1) time..


    @staticmethod
    def is_in_bounds(state, pos: tuple) -> bool:
        """Checks if a target coordinate is within the Pygame grid."""
        x, y = pos
        return 0 <= x < state.grid_size and 0 <= y < state.grid_size

  
    @staticmethod 
    def get_distance(state : GameState, id1 : int, id2 : int) -> int:
        return int(np.sum(np.abs(state.positions[id1] - state.positions[id2])))  ## Note that state.positions is also an array which cotains an inner array, so the subtraction takes place like in matrices
    
    @staticmethod
    def get_hp_ratio(state : GameState, agent_id : int):
        return (state.hp[agent_id] / state.max_hp[agent_id])

    @staticmethod
    def get_stamina_ratio(state : GameState, agent_id : int):
        return (state.stamina[agent_id] / state.max_stamina[agent_id])
    
    ### POSITION HELPERS
    @staticmethod 
    def get_position(state : GameState, agent_id : int):
        return state.positions[agent_id]
    
    @staticmethod 
    def set_position(state : GameState, agent_id, position : Tuple[int, int]):

        state.positions[agent_id] = np.array(position)


    ### COOLDOWN HELPERS

    @staticmethod
    def is_skill_ready(state : GameState, agent_id : int, action_type : ActionTypes):
        skill_idx = state.ACTION_TO_INDEX_MAP[action_type]
        return state.cooldowns[agent_id, skill_idx] == 0 


    @staticmethod
    def update_cooldown(state : GameState) -> None:
   ## Note that that 1 is broadcasted to match the whole size of state.cooldowns .. (num_agents, num_skills).
        state.cooldowns = np.maximum(0, state.cooldowns - 1)   # Normal max wont work for numpy arrays as state.cooldowns[agent_id] is also an array of 3 skill coolwdowns


    ### STAMINA HELPERS 

    @staticmethod 
    def has_stamina(state : GameState, agent_id : int, stamina_cost : float) -> bool:
        return state.stamina[agent_id] >= stamina_cost

    @staticmethod 
    def consume_stamina(state : GameState, agent_id : int, stamina_cost : float) -> None:

        state.stamina[agent_id] = max(0.0, state.stamina[agent_id] - stamina_cost)

    
    ### TEAM WIN HELPERS 

    @staticmethod
    def is_terminal(state : GameState) -> bool:
        """
        Checks if the episode should end.
        Returns True if either the Boss is dead, or ALL Heroes are dead.
        """
        # np.any() checks if AT LEAST ONE value in the masked array is True
        heroes_alive = np.any(state.hp[state.team_masks[Teams.HEROES]] > 0)
        monsters_alive = np.any(state.hp[state.team_masks[Teams.MONSTERS]] > 0)
        
        return not (heroes_alive and monsters_alive)   # Works three times - If either one is False (AND returns false) and also when both are false..

    
    @staticmethod 
    def get_winning_team(state : GameState) :

        heroes_alive = np.any(state.hp[state.team_masks[Teams.HEROES]] > 0)
        monsters_alive = np.any(state.hp[state.team_masks[Teams.MONSTERS]] > 0)

        if not heroes_alive :
            return Teams.MONSTERS

        else :
            return Teams.HEROES

        
    ### RANGE QUEURIES 

    @staticmethod 
    def get_agents_in_range(state : GameState, source_agent_id : int, min_range : int, max_range : int):

        agent_ids = StateOperations.get_alive_agent_ids(state)
        source_position = state.positions[source_agent_id]
        agent_positions = state.positions[agent_ids]

        distances = np.sum(np.abs(agent_positions - source_position), axis = 1)

        range_mask = (distances >= min_range) & (distances <= max_range)

        return agent_ids[range_mask] ## Boolean indexing..

    
    @staticmethod 
    def get_allies_in_range(state : GameState, source_agent_id : int, min_range : int, max_range : int):
        
        ally_ids = StateOperations.get_alive_allies(state, source_agent_id)
        source_position = state.positions[source_agent_id]
    
        ally_positions = state.positions[ally_ids]  ## Returns all the arrays of that many ally ids.. that is shape = (num_allies, 2) 

        distances = np.sum(np.abs(ally_positions - source_position), axis = 1)  ## Although the sizes dont match, we have ally_pos = (2,2) and source to be (2,) which is converted to (1,2) by numpy. And that single thing is copied virtually (not physically) that is broadcasted to match all the enemies .. and then subtraction takes place.
        ## Also that np.abs takes place along columns..it takes the absolute value that is makes negative .. positive and np.sum -  that is for each row we calculate the total of that row.

        range_mask = (distances >= min_range) & (distances <= max_range)

        return ally_ids[range_mask] ## Boolean indexing..



    @staticmethod 
    def get_enemies_in_range(state : GameState, source_agent_id : int, min_range : int, max_range : int):
        
        source_position = state.positions[source_agent_id] 
        enemy_ids = StateOperations.get_alive_enemies(state, source_agent_id)
        enemy_positions = state.positions[enemy_ids]  ## Returns all the arrays of that many enemy ids.. that is shape = (num_enemies, 2) 

        distances = np.sum(np.abs(enemy_positions - source_position), axis = 1)  ## Although the sizes dont match, we have enemy_pos = (2,2) and source to be (2,) which is converted to (1,2) by numpy. And that single thing is copied virtually (not physically) that is broadcasted to match all the enemies .. and then subtraction takes place.
        ## Also that np.abs takes place along columns.. With np.sum that is for each row we calculate the total of that row.

        range_mask = (distances >= min_range) & (distances <= max_range)

        return enemy_ids[range_mask] ## Boolean indexing..




    ### Regarding unique tiles visited
        
    @staticmethod
    def is_unique(state : GameState, agent_id : int, pos : Tuple[int, int]):

        x,y = pos 
        team = state.teams[agent_id]

        if state.team_visited_tiles[team, x, y] == True :
            return False 

        return True 

        
    @staticmethod
    def update_team_visited_tiles(state : GameState, agent_id : int, pos : Tuple[int, int]):

        x,y = pos
        team = state.teams[agent_id]

        if StateOperations.is_unique(state, agent_id, pos):
            state.team_visited_tiles[team, x, y] = True 


    @staticmethod 
    def recover_stamina(state : GameState):

        state.stamina = np.minimum(state.max_stamina, state.stamina + state.recovery_rates)


    @staticmethod
    def normalize_dims(state : GameState, agent_id : int):
        x,y = state.positions[agent_id]
        return x / state.grid_size, y / state.grid_size

            

    ### Action MASK
    @staticmethod
    def get_action_mask(state: GameState, agent_id: int) -> np.ndarray:
        """
        Returns a boolean array of shape (8,) where False indicates an invalid action.
        """
        mask = np.ones(8, dtype=np.bool_)
        
        # Dead agents cannot execute any actions
        if state.hp[agent_id] <= 0:
            mask.fill(False)
            return mask

        current_pos = state.positions[agent_id]
        occupied_positions = StateOperations.get_occupied_positions(state, exclude_id=agent_id)

        # 1. MOVEMENT BOUNDARY & COLLISION VALIDATION
        # UP (Checks grid top border and cell occupancy)
        if current_pos[1] <= 0 or (current_pos[0], current_pos[1] - 1) in occupied_positions:
            mask[0] = False # ActionTypes.UP
            
        # DOWN (Checks grid bottom border)
        if current_pos[1] >= state.grid_size - 1 or (current_pos[0], current_pos[1] + 1) in occupied_positions:
            mask[1] = False # ActionTypes.DOWN
            
        # LEFT (Checks grid left border)
        if current_pos[0] <= 0 or (current_pos[0] - 1, current_pos[1]) in occupied_positions:
            mask[2] = False # ActionTypes.LEFT
            
        # RIGHT (Checks grid right border)
        if current_pos[0] >= state.grid_size - 1 or (current_pos[0] + 1, current_pos[1]) in occupied_positions:
            mask[3] = False # ActionTypes.RIGHT

        # 2. STAMINA AND COOLDOWN RESOURCE VALIDATION
        # Index 5 maps to BASIC, 6 to UTILITY, 7 to ULTIMATE
        for act_idx in [5, 6, 7]:
            stamina_cost = state.stamina_cost[agent_id, act_idx]
            cooldown_remaining = state.cooldowns[agent_id, act_idx - 5] # Cooldown maps to [0, 1, 2]

            if state.stamina[agent_id] < stamina_cost or cooldown_remaining > 0:
                mask[act_idx] = False

        return mask


    
    @staticmethod 
    def state_snapshot(state : GameState):
        return copy.deepcopy(state)  ## a deep copy to get different memory references for arrays inside  ....