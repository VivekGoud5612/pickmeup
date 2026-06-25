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
        self.invalid_action_penalty = -0.1 
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

    def _formation_score(self, state : GameState):  ## Measure how well the formation of heroes is 

        heroes_mask = stateops.get_team_alive_mask(state, Teams.HEROES)
        active_positions = state.positions[heroes_mask]

        if len(active_positions) <= 1: # If there are 0 or 1 agents alive then there is no meaning to calculate the formation score
            return 0.0
        
        centroid = np.mean(active_positions, axis = 0)  # Does this mean that we are summing up all the pos (of shape [2,]) of all alive hero agents (axis = 0) and averaging over them ? Centroid  shape [2,].
        distances = np.sum(np.abs(active_positions - centroid), axis = 1)   # Axis 1 means that each position array subtracts and np.abs check if there are no negatives..And np.sum (of an array of size lets say (3,2) along  axis 1 would mean add each row, so end result is (3,1))
        
        return -float(np.mean(distances))  ## This way we can see if the group is closer or not. If the negative is verylarge that means that the agents are spread out. This automatically affects the reward, so it tries to minimize that closer to 0, which would make the formation close..

    
    def _dealer_position_score(self, state : GameState) -> float : # Reward if dealer is at a safe range than the boss and how much damage he did to the boss
        
        if state.hp[self.dealer_id] <= 0 or state.hp[self.boss_id] <= 0:
            return 0.0  #There is no meaning to calculate the reward of distance if both of them are dead.

        distance = stateops.get_distance(state, self.dealer_id, self.boss_id)
        return 1.0 if (2 <= distance <= 4) else -float(abs(distance - 2.5))  ## If the distance is out of that range then dealer gets a negative reward

    def _healer_safety_score(self, state : GameState) -> float:

        if state.hp[self.healer_id] <= 0:
            return 0.0
        score = 0.0

        if state.hp[self.boss_id] > 0:
            score += stateops.get_distance(state, self.healer_id, self.boss_id) * 0.5 ## We add this to score as we want to maximise this, that is maximise the distance    

        if state.hp[self.tank_id] > 0:
            score -= stateops.get_distance(state, self.healer_id, self.tank_id) * 0.5 ## We want to minimize this as want the distance between tank and healer to be low...

        return float(score)

    def _tanker_protection_score(self, state : GameState) -> float:  ## POsitioning of tank between dealer, healer and the boss...

        if state.hp[self.tank_id] <= 0 or state.hp[self.boss_id] <= 0:
            return 0.0

        score = 0.0 
        tank_to_boss = stateops.get_distance(state, self.tank_id, self.boss_id)

        for squishy_id in [self.healer_id, self.dealer_id]:

            if state.hp[squishy_id] > 0:
                squishy_to_boss = stateops.get_distance(state, squishy_id, self.boss_id)  ## Distance between squishy agents and boss needs to be large
                if squishy_to_boss < tank_to_boss: ## If tank is near to boss than those 2
                    score += 2.0
                else:
                    score -= 0.5

        return float(score)

    def _boss_attack_weak_score(self, state : GameState): ## THis is just so that boss attacks weaker sections of the game.. or the ones which changing the games course like healer

        if state.hp[self.boss_id] <= 0 or not np.any(state.hp[stateops.get_team_alive_mask(state, Teams.HEROES) > 0]):
            return 0.0 

        score = 0.0
        boss_to_tank = stateops.get_distance(state, self.tank_id, self.boss_id)

        for squishy_id in [self.healer_id, self.dealer_id]:
            boss_to_squishy = stateops.get_distance(state, squishy_id, self.boss_id)
            if boss_to_tank > boss_to_squishy:  # Boss should target weaker sections
                score += 3.0

            else:
                score -= 0.5

        return float(score)

    def _get_handcrafted_potential(self, state : GameState) -> Tuple[float, float]:

        potentials = np.zeros(state.num_agents, dtype = np.float32)

        alive_heroes = len(stateops.get_alive_team_agent_ids(state, Teams.HEROES))

        hero_hp = self._hero_hp_score(state)
        boss_hp_penalty = -self._boss_hp_score(state)   ## a negative score for heroes as boss shouldnt be alive... 
        formation = self._formation_score(state)  # The amount of spread(is negative and agent minimizes that)...
        hero_stamina = self._hero_stamina_score(state)

        boss_hp = self._boss_hp_score(state)
        hero_hp_penalty = -hero_hp  # a penalty for boss as heroes are still alive
        boss_stamina = self._boss_stamina_score(state)

        dealer_pos = self._dealer_position_score(state)
        healer_safe = self._healer_safety_score(state)
        tank_prot = self._tanker_protection_score(state)

        boss_squish = self._boss_attack_weak_score(state)

        ## Hero potential contains a lot of things summed weighted
        shared_hero_pot = (hero_hp * 10.0) + (alive_heroes * 2.0) + (formation * 0.5) + (hero_stamina * 0.5) + (boss_hp_penalty * 10.0)

        potentials[self.dealer_id] = shared_hero_pot + dealer_pos 
        potentials[self.tank_id] = shared_hero_pot + tank_prot 
        potentials[self.healer_id] = shared_hero_pot + healer_safe
        potentials[self.boss_id] = (boss_hp * 10.0) + (hero_hp_penalty * 10.0) + (boss_stamina * 1.5) + boss_squish 

        return potentials


    def _combat_reward_for_dealer(self, state : GameState, rewards : np.ndarray):

        damage = state.damage_dealt[self.dealer_id]
        stamina_spent = state.stamina_spent[self.dealer_id]

        if damage > 0:
            efficiency = damage / (stamina_spent + 1.0)
            rewards[self.dealer_id] += (damage * 0.5) + (efficiency * 2.0)

    def _combat_reward_for_tank(self, state : GameState, rewards : np.ndarray):

        rewards[self.tank_id] += state.damage_dealt[self.tank_id]

        blocked_dam = state.damage_reduction_by_block[self.tank_id]
        null_dam = state.damage_reduction_by_nullification[self.tank_id]

        if blocked_dam > 0:
            rewards[self.tank_id] += blocked_dam * 1.5

        if null_dam > 0:
            rewards[self.tank_id] += null_dam * 2.0

    def _combat_reward_for_healer(self, state : GameState, rewards : np.ndarray):

        effective_heal = state.effective_heal[self.healer_id]
        rewards[self.healer_id] += effective_heal * 0.8

        if effective_heal == 0 and state.stamina_spent[self.healer_id]:  # wasting stamina on 0 effective heal, maybe healer used it on agents with full hp
            rewards[self.healer_id] -= 1.0

    def _combat_reward_for_boss(self, state : GameState, rewards : np.ndarray):

        effective_heal = state.effective_heal[self.boss_id]
        rewards[self.boss_id] += effective_heal * 0.8

        if effective_heal == 0 and state.stamina_spent[self.boss_id]:  # wasting stamina on 0 effective heal, maybe healer used it on agents with full hp
            rewards[self.boss_id] -= 1.0

        rewards[self.boss_id] += state.damage_dealt[self.boss_id]  # I guess there is no need for efficiency check, because boss is expected to be something with high stamina

    def calculate_decomposed_reward(self, old_state : GameState, new_state : GameState, phase : int, old_values : np.ndarray = None, new_values : np.ndarray  = None):  ## This contains the phase based reward calculation or more specifically all three reward calculations

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
                combat_rewards[idx] += 0.5  ## Boost intrinsic motivation for exploring new states.
        

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

