from engine.environment.state import GameState
from engine.environment.state_ops import StateOperations as stateops
from typing import Dict,List,Any
from enum import IntEnum

class ActionTypes(IntEnum):  ## Now there is no need for action map or reverse action map...
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    WAIT = 4
    BASIC = 5
    UTILITY = 6
    ULTIMATE = 7

class ActionHandler:

    @staticmethod 
    def _apply_damage(state : GameState, attacker_id : int, target_id : int, skill_mult : float, stamina_cost : float, ignore_defence : bool = False):  # The core function which calculates and applies damage

        state.stamina_spent[attacker_id] += stamina_cost 
        stateops.consume_stamina(state, attacker_id, stamina_cost)

        if state.is_invincible[target_id] :
            attack_power = 0
            state.damage_reduction_by_nullification[target_id] += state.strengths[attacker_id]   # Invincibility also counts as a block

        else:
            attack_power = state.strengths[attacker_id] * skill_mult 
        
        base_defence = state.defences[target_id]
        effective_defence = 0.0 if ignore_defence else base_defence
                
        mitigation = 100.0 / (100.0 + base_denfence)   # For combat reward calculation...
        base_damage_without_block = min(state.hp[target_id], attack_power * mitigation) ## IF the actual health is 10 and boss fires a 100 attack on this, then the damage would only be 10, so it helps preserve large damage attacks for other agents.    

        if not ignore_defence and state.is_blocking[target_id]:
            effective_defence *= 2

        mitigation_multiplier = 100.0 / (100.0 + effective_defence)
        actual_damage = min(state.hp[target_id], attack_power * mitigation_multiplier)

        state.damage_dealt[attacker_id] += actual_damage  ## If the health is less than the damage... we can simply write prev_hp as well..
        state.damage_taken[target_id] += actual_damage

        hp_loss_with_block = max(state.hp[target_id], state.hp[target_id] - actual_damage) ## IF the actual health is 10 and boss fires a 100 attack on this, then the damage would only be 10, so it helps preserve large damage attacks for other agents.    

        state.hp[target_id] = max(0.0, state.hp[target_id] - actual_damage)

        #state.damage_dealt[attacker_id] += min(actual_damage, prev_hp - state.hp[target_id])   ## If the health is less than the damage... we can simply write prev_hp as well..

        state.damage_reduction_by_block[target_id] = hp_loss_without_block - hp_loss_with_block 


    
    ## First Attacking actions are resolved
    @staticmethod 
    def resolve_basic_attack(state : GameState, agent_id : int):

        skill_mult = state.skill_multiplier[agent_id, ActionTypes.BASIC]  # as we are taking the skills from here , 
        min_range = state.min_ranges[agent_id, ActionTypes.BASIC]
        max_range = state.max_ranges[agent_id, ActionTypes.BASIC]
        stamina_cost = state.stamina_cost[agent_id, ActionTypes.BASIC]

        enemies_in_range = stateops.get_enemies_in_range(state, agent_id, min_range, max_range)

        if len(enemies_in_range) == 0:
            return

        target_id = enemies_in_range[0]  # Standard way to take the very first enemy who is more closer 
        ActionHandler._apply_damage(state, agent_id, target_id, skill_mult, stamina_cost)  

    @staticmethod 
    def resolve_pierce(state : GameState, agent_id : int):
        """Dealer Utility: Raw execution that ignores armor entirely."""
        skill_mult = state.skill_multiplier[agent_id, ActionTypes.UTILITY]  # as we are taking the skills from here , 
        min_range = state.min_ranges[agent_id, ActionTypes.UTILITY]
        max_range = state.max_ranges[agent_id, ActionTypes.UTILITY]
        stamina_cost = state.stamina_cost[agent_id, ActionTypes.UTILITY]
        
        enemies = stateops.get_enemies_in_range(state, agent_id, min_range = min_range, max_range = max_range)
        if len(enemies) > 0:
            ActionHandler._apply_damage(state, agent_id, enemies[0], skill_mult, stamina_cost, ignore_defence=True)

        
    @staticmethod 
    def resolve_special(state : GameState, agent_id : int):
        
        skill_mult = state.skill_multiplier[agent_id, ActionTypes.ULTIMATE]  # as we are taking the skills from here , 
        min_range = state.min_ranges[agent_id, ActionTypes.ULTIMATE]
        max_range = state.max_ranges[agent_id, ActionTypes.ULTIMATE]
        stamina_cost = state.stamina_cost[agent_id, ActionTypes.ULTIMATE]

        enemies = stateops.get_enemies_in_range(state, agent_id, min_range = min_range, max_range = max_range)
        if len(enemies) > 0:
            ActionHandler._apply_damage(state, agent_id, enemies[0], skill_mult, stamina_cost)
         

    @staticmethod
    def resolve_aoe(state : GameState, agent_id : int):

        skill_mult = state.skill_multiplier[agent_id, ActionTypes.ULTIMATE]  # as we are taking the skills from here , 
        min_range = state.min_ranges[agent_id, ActionTypes.ULTIMATE]
        max_range = state.max_ranges[agent_id, ActionTypes.ULTIMATE]
        stamina_cost = state.stamina_cost[agent_id, ActionTypes.ULTIMATE]

        enemies = stateops.get_enemies_in_range(state, agent_id, min_range = min_range, max_range = max_range)

        stateops.consume_stamina(state, agent_id, stamina_cost)  # As in that loop boss stammina is being consumed for all enemies..

        effective_skill_mult = skill_mult / len(enemies)   ## to divie a large amount of damage among number of enemies in range (equally for now), after we can maybe see if we can do that based on distance..
        # As we cannot change the strength, we change the skill multiplier...

        for enemy in enemies:  # AOE so we attack all the agents inside range...
            ActionHandler._apply_damage(state, agent_id, enemy, effective_skill_mult, 0.0)



    ## DEFENCE AND SUPPORT ACTIONS
    @staticmethod
    def resolve_invincible(state : GameState, agent_id):  ## A single time action which makes all the incoming attack to 0.. consumes a lot of Tanks stamina
        state.is_invincible[agent_id] = True   # TANK ULTIMATE
        stamina_cost = state.stamina_cost[agent_id, ActionTypes.ULTIMATE]
        state.stamina_spent[agent_id] += stamina_cost
        stateops.consume_stamina(state, agent_id, stamina_cost)

    @staticmethod 
    def resolve_block(state : GameState, agent_id : int):
        state.is_blocking[agent_id] = True
        stamina_cost = state.stamina_cost[agent_id, ActionTypes.UTILITY]
        state.stamina_spent[agent_id] += stamina_cost
        stateops.consume_stamina(state, agent_id, stamina_cost)

    @staticmethod
    def resolve_heal(state, agent_id: int) -> None:
        """Healer Utility: Targets and scales single target healing to the lowest health ally."""

        skill_mult = state.skill_multiplier[agent_id, ActionTypes.UTILITY]  # as we are taking the skills from here , 
        min_range = state.min_ranges[agent_id, ActionTypes.UTILITY]
        max_range = state.max_ranges[agent_id, ActionTypes.UTILITY]
        stamina_cost = state.stamina_cost[agent_id, ActionTypes.UTILITY]

        allies = stateops.get_allies_in_range(state, agent_id, min_range = min_range, max_range = max_range)

        if len(allies) == 0:
            return
            
        # Compute exact relative HP matrices via vectorized calculations
        hp_ratios = state.hp[allies] / state.max_hp[allies]   ## Allies is an array of agent_ids, so numpy calculates the ratio respectively and stacks them inside another array
        target_id = allies[np.argmin(hp_ratios)]  # We take the id of minimum hp ratio..
        
        heal_value = state.strengths[agent_id] * skill_mult

        ## Calculating effective heal...
        old_hp = state.hp[target_id]
        state.hp[target_id] = min(state.max_hp[target_id], state.hp[target_id] + heal_value)
        effective_heal = state.hp[target_id] - old_hp

        state.effective_heal[agent_id] += effective_heal

        state.stamina_spent[agent_id] += stamina_cost  ## For reward calculation... so needed
        stateops.consume_stamina(state, agent_id, stamina_cost)

    @staticmethod
    def resolve_all_heal(state, agent_id: int) -> None:
        """Healer Ultimate: Dispatches bulk restoration values across the entire field group."""
        
        skill_mult = state.skill_multiplier[agent_id, ActionTypes.ULTIMATE]  # as we are taking the skills from here , 
        min_range = state.min_ranges[agent_id, ActionTypes.ULTIMATE]
        max_range = state.max_ranges[agent_id, ActionTypes.ULTIMATE]
        stamina_cost = state.stamina_cost[agent_id, ActionTypes.ULTIMATE]

        allies = stateops.get_allies_in_range(state, agent_id, min_range = min_range, max_range = max_range)

        if len(allies) == 0:
            return
            
        heal_value = state.strengths[agent_id] * skill_mult

        ## Effective heal for all heal as well
        old_hps = state.hp[allies]
        state.hp[allies] = np.minimum(state.max_hp[allies], state.hp[allies] + heal_value)
        effective_heals = state.hp[allies] - old_hps ## shape (num_allies,)

        effective_heal = np.sum(effective_heals, axis = 0)

        state.effective_heal[agent_id] += effective_heal

        state.stamina_spent[agent_id] += stamina_cost 
        stateops.consume_stamina(state, agent_id, stamina_cost)

    @staticmethod
    def resolve_regenerate(state, agent_id: int) -> None:
        """Boss Utility: Direct self restoration loop processing."""
        skill_mult = state.skill_multiplier[agent_id, ActionTypes.UTILITY]  # as we are taking the skills from here , 
        stamina_cost = state.stamina_cost[agent_id, ActionTypes.UTILITY]

        heal_value = state.strengths[agent_id] * skill_mult

        old_hp = state.hp[agent_id]
        state.hp[agent_id] = min(state.max_hp[agent_id], state.hp[agent_id] + heal_value)
        effective_heal = state.hp[agent_id] - old_hp 

        state.effective_heal[agent_id] += effective_heal

        state.stamina_spent[agent_id] += stamina_cost
        stateops.consume_stamina(state, agent_id, stamina_cost)

       
