from typing import List, Dict, Any 
from engine.environment.state_ops import StateOperations as stateops
from engine.environment.state import GameState 
from engine.actions.action_handler import ActionHandler
from engine.actions.action_handler import ActionTypes 
from engine.agents.agent_data import AgentRole
import numpy as np


class ActionSequencer:

    @staticmethod
    def resolve_step(state : GameState, actions : List[ActionTypes]) -> None:  ### This is function to resolve all actions in phases...Also that actions is a numpy array with size (num_envs, num_agents).. althoug we get actions separately per agent from actor, we append that and send it the sequencer. 
        
        ## Actions are basically a list of Action Types as we changed them back in base_agent , where if we get an action we map it to our map dictionary...
        
        state.is_blocking.fill(False)  ## We set all the blockings to False at each new step, because the tank blocks once per step and has a cooldown. This needs to reset or else the tank stays on block the whole episode...
        state.damage_dealt.fill(0.0)
        state.effective_healing.fill(0.0)
        state.damage_reduction_by_block.fill(0.0)
        state.damage_reduction_by_nullification.fill(0.0)
        state.stamina_spent.fill(0.0)
        state.damage_taken.fill(0.0)

        ## mutatable_action_copy = np.array([action_type.value for action_type in actions]) ## Size same as actions which is (num_angents,). And action_type.value would give the necessary integer..
        
        processed = np.zeros(len(actions), dtype = bool)

        alive_agents = stateops.get_alive_agent_ids(state)
        np.random.shuffle(alive_agents)   # the reason for that random shuffle was because if tank is first everytime and it chooses to move to a direction, then it would be unfair for other agents wanting to move there.In a 2D grid world this is the best without continuos actions or states.

        ## PHASE 1... DEFENCE and HEALING.. we resolve both premovement actions here ...
        for agent_id in alive_agents:

            if not stateops.is_alive(state, agent_id) or processed[agent_id]:
                continue

            action = actions[agent_id]
            role = state.roles[agent_id]

            if role == AgentRole.TANK :  # Tank and a utility skills... block
                
                if action == ActionTypes.UTILITY:
                    ActionHandler.resolve_block(state, agent_id)
                    processed[agent_id] = True 

                elif action == ActionTypes.ULTIMATE:
                    ActionHandler.resolve_invincible(state, agent_id)
                    processed[agent_id] = True
            
            elif role == AgentRole.HEALER:
                if action == ActionTypes.UTILITY:  # That is if healer used HEAL.. and note that Healers basic skills is a simple magic attack..resolved later
                    ActionHandler.resolve_heal(state, agent_id)
                    processed[agent_id] = True

                elif action == ActionTypes.ULTIMATE: ## As healers ultimate is also a pre movement skill which is not combat related..
                    ActionHandler.resolve_all_heal(state, agent_id)
                    processed[agent_id] = True

            elif role == AgentRole.BOSS and action == ActionTypes.UTILITY: ## as boss also has a pre movement skill which is not combat related which is regenerate.. utility skill marked as 6 in the action map
                ActionHandler.resolve_regenerate(state, agent_id)
                processed[agent_id] = True

        ## PHASE 2... Movement... before attack... this promotes evading and this has a higher priority because if some agent tried to evade and got hit even then that doesnt make sense
        for agent_id in alive_agents:

            if not stateops.is_alive(state, agent_id) or processed[agent_id]:
                continue 
            
            action = actions[agent_id]
            
            if action in [ActionTypes.UP, ActionTypes.DOWN, ActionTypes.LEFT, ActionTypes.RIGHT]:
                ActionSequencer._attempt_move(state, agent_id, action)
                processed[agent_id] = True

            elif action == ActionTypes.WAIT:
                processed[agent_id] = True
            
        
        ## PHASE 3 ... Attack , contains all the attacks here including healers basic magic attack, boss AOE and other attacks.
        for agent_id in alive_agents:
             
            if not stateops.is_alive(state, agent_id) or processed[agent_id]: ## Althogh there is only no chance of anything dying before this so no need for this but its ok..
                continue

            action = actions[agent_id]
            role = state.roles[agent_id] 

            if action == ActionTypes.BASIC: ### As all the roles have this... a simple damage taking basic attack
                ActionHandler.resolve_basic_attack(state, agent_id)
                processed[agent_id] = True

            elif role == AgentRole.DEALER:  ## We resolve both pierce and special of dealer here...
                
                if action == ActionTypes.UTILITY: ## If dealer chose pierce 
                    ActionHandler.resolve_pierce(state, agent_id)
                    processed[agent_id] = True

                elif action == ActionTypes.ULTIMATE: ## If dealer chose his ultimate speical attack 
                    ActionHandler.resolve_special(state, agent_id)
                    processed[agent_id] = True


            elif role == AgentRole.BOSS and action == ActionTypes.ULTIMATE: ## Boss ultimate AOE
                ActionHandler.resolve_aoe(state, agent_id) 
                processed[agent_id] = True

        stateops.update_cooldowns(state)
        stateops.recover_stamina(state)

    @staticmethod 
    def _attempt_move(state : GameState, agent_id : int, direction : ActionTypes) -> None:  # So I guess movement related actions are done so we just need to resole all combat related actions..
        ## We validate the position before moving the agent there. Like if a agent occupies that then there could be a problem..

        current_pos = state.positions[agent_id]  ##  np.array of shape (2,)
        new_pos = np.copy(current_pos)  #Initially we copy the current pos and then update that and see if that is valid, else fallback to current pos

        if direction == ActionTypes.UP : new_pos[1] -= 1 ## Up , we reduce the row by one 
        if direction == ActionTypes.DOWN : new_pos[1] += 1 ## Down 
        if direction == ActionTypes.LEFT : new_pos[0] -= 1 # Left so we reduce the column number ny 1
        if direction == ActionTypes.RIGHT : new_pos[0] += 1 #RIGHT 

        if not stateops.is_in_bounds(state, tuple(new_pos)): # We check if the new updated position is out of bounds ..
            return 'Hit a wall, move failed and position is the same'

        occupied_pos = stateops.get_occupied_positions(state, exclude_id = agent_id) ## We do not want to count the agents current position, so we exlcude that 
        if tuple(new_pos) in occupied_pos: ## Lookup O(1) as our occupied pos is a set()
            return 'Hit another agent, move failed and position is the same'

        state.positions[agent_id] = new_pos ## If none of that happens then we can safely update the agents positions..

        if stateops.is_unique(state, agent_id, new_pos):
            stateops.update_unique_tiles(state, agent_id, new_pos)
        

            
