from typing import Dict, Tuple, Any 
import math 
import numpy as np 
from engine.utils.enums import AgentRole, Teams, ActionTypes, AgentID
from engine.environment.state import GameState 
from engine.environment.state_ops import StateOperations as stateops 

class RewardCalculator:
    
    def __init__(self, gamma : float = 0.80):

        self.gamma = gamma 

        self.time_step_penalty = -0.1 
        self.win_bounty = 100.0 
        self.death_penalty = -0.50 

        self.tank_id = AgentID.TANK 
        self.dealer_id = AgentID.DEALER 
        self.healer_id = AgentID.HEALER 
        self.boss_id = AgentID.BOSS 

    
    ## HANDCRAFTED SCORES (Not Dynamic) --- Instead of plain hp difference and distance, we measure many things here
    def _hero_hp_score(self, state : GameState) -> float: ## A simple calculation of mean of hp ratios...

        heroes_mask = stateops.get_team_alive_mask(state, Teams.HEROES)

        if not np.any(state.hp[heroes_mask] > 0):
            return 0.0
        return float(np.mean(state.hp[heroes_mask] / state.max_hp[heroes_mask]))

    def _boss_hp_score(self, state : GameState):
        return float(state.hp[self.boss_id] / state.max_hp[self.boss_id])

    def _hero_stamina_score(self, state : GameState):  # Stamina score for only the heroes

        heroes_mask = state.teams[Teams.HEROES]
        return float(np.mean(state.stamina[heroes_mask] / state.max_stamina[heroes_mask]))  ## There is no need for mask here, because we are calculating the stamina of all in potentials... But I guess we can separate the staminas out..
        

    def _boss_stamina_score(self, state : GameState):
        return float(state.stamina[self.boss_id] / state.max_stamina[self.boss_id])

    def _get_handcrafted_potential(self, state : GameState) -> Tuple[float, float]:

        potentials = np.zeros(state.num_agents, dtype = np.float32)

        alive_heroes = len(stateops.get_alive_team_agent_ids(state, Teams.HEROES))

        hero_hp = self._hero_hp_score(state)
        boss_hp_penalty = -self._boss_hp_score(state)   ## a negative score for heroes as boss shouldnt be alive... 
        hero_stamina = self._hero_stamina_score(state)

        boss_hp = self._boss_hp_score(state)
        hero_hp_penalty = -hero_hp  # a penalty for boss as heroes are still alive
        boss_stamina = self._boss_stamina_score(state)

        ## Hero potential contains a lot of things summed weighted
        shared_hero_pot = (hero_hp * 10.0) + (hero_stamina * 1.5) + (boss_hp_penalty * 10.0)

        potentials[self.dealer_id] = shared_hero_pot
        potentials[self.tank_id] = shared_hero_pot 
        potentials[self.healer_id] = shared_hero_pot
        potentials[self.boss_id] = (boss_hp * 10.0) + (hero_hp_penalty * 10.0) + (boss_stamina * 1.5)

        return potentials


    def _combat_reward_for_dealer(self, state : GameState, rewards : np.ndarray):

        damage = state.damage_dealt[self.dealer_id]

        if damage > 0:
            rewards[self.dealer_id] += (damage * 1.0)

    def _combat_reward_for_tank(self, state : GameState, rewards : np.ndarray):

        rewards[self.tank_id] += state.damage_dealt[self.tank_id]

        blocked_dam = state.damage_reduction_by_block[self.tank_id]
        null_dam = state.damage_reduction_by_nullification[self.tank_id]

        if blocked_dam > 0:
            rewards[self.tank_id] += blocked_dam * 1.0

        if null_dam > 0:
            rewards[self.tank_id] += null_dam * 1.0  ## same weights as others so that the weights are dependant on neural network..

    def _combat_reward_for_healer(self, state : GameState, rewards : np.ndarray):

        rewards[self.healer_id] += state.damage_dealt[self.healer_id]  ## As healer also has a attack spell (basic we can do that)

        effective_heal = state.effective_heal[self.healer_id]
        rewards[self.healer_id] += effective_heal * 1.0

        #if effective_heal == 0 and state.stamina_spent[self.healer_id]:  # wasting stamina on 0 effective heal, maybe healer used it on agents with full hp
            #rewards[self.healer_id] -= 1.0

    def _combat_reward_for_boss(self, state : GameState, rewards : np.ndarray):

        effective_heal = state.effective_heal[self.boss_id]
        rewards[self.boss_id] += effective_heal * 1.0

        rewards[self.boss_id] += state.damage_dealt[self.boss_id]  # I guess there is no need for efficiency check, because boss is expected to be something with high stamina

    def calculate_decomposed_reward(self, old_state : GameState, new_state : GameState):  ## This contains the phase based reward calculation or more specifically all three reward calculations

        num_agents = new_state.num_agents
        
        terminal_rewards = np.zeros(num_agents, dtype = np.float32) 
        combat_rewards = np.zeros(num_agents, dtype = np.float32)  ## BAsed on how well the impact of the agents attack took place (need to write again)


        old_handcrafted = self._get_handcrafted_potential(old_state)
        new_handcrafted = self._get_handcrafted_potential(new_state)

        ### Combat based reward calculation.. checking if that action was good or not
        for idx in range(num_agents):
            if old_state.hp[idx] <= 0:
                continue  ## Dead agents are skipped, as there is no need for the whole reward calculation when the contribution is zero.

            role = old_state.roles[idx]

            if role == AgentRole.DEALER:
                self._combat_reward_for_dealer(new_state, combat_rewards)

            elif role == AgentRole.TANK:
                self._combat_reward_for_tank(new_state, combat_rewards)

            elif role == AgentRole.HEALER:
                self._combat_reward_for_healer(new_state, combat_rewards)

            elif role == AgentRole.BOSS:
                self._combat_reward_for_boss(new_state, combat_rewards)

            if new_state.exploration_bonus_triggered[idx] > 0:
                combat_rewards[idx] += 0.05  ## Boost intrinsic motivation for exploring new states.

        combat_rewards += self.time_step_penalty

        ### Terminal rewards
        heroes_mask = new_state.team_masks[Teams.HEROES]

        for idx in range(num_agents):
            was_alive = old_state.hp[idx] > 0
            is_alive = new_state.hp[idx] > 0

            if was_alive and not is_alive:
                terminal_rewards[idx] += self.death_penalty

                if idx == self.boss_id:
                    living_heroes = stateops.get_alive_team_agent_ids(new_state, Teams.HEROES)   ##Boss defeated so heroes get some win bounty

                    for hero_id in living_heroes:
                        terminal_rewards[hero_id] += self.win_bounty 

                else:
                    terminal_rewards[self.boss_id] -= self.death_penalty   # IF heroes dead then boss gets good reward

        return {
            'old_handcrafted' : old_handcrafted, ## BEfore old_hand (1,) ... now oldhandcrafted (4, )
            'new_handcrafted' : new_handcrafted, 
            'combat_rewards' : combat_rewards,
            'terminal_rewards' : terminal_rewards,
        }

