from typing import Dict, Tuple, Any 
import math 
import numpy as np 
from engine.actions.action_handler import ActionTypes
from engine.agents.agent_data import AgentRole, Teams 
from engine.environment.state import GameState 
from engine.environment.state_ops import StateOperations as stateops 

class RewardCalculator:
    
    def __init__(self, gamma : float = 0.80):

        self.gamma = gamma 

        self.time_step_penalty = -0.1 
        self.invalid_action_penalty = -0.1 
        self.win_bounty = 100.0 
        self.death_penalty = -0.50 

        self.tank_id = AgentRole.TANK 
        self.dealer_id = AgentRole.DEALER 
        self.healer_id = AgentRole.HEALER 
        self.boss_id = AgentRole.BOSS 

    
    ## HANDCRAFTED SCORES (Not Dynamic) --- Instead of plain hp difference and distance, we measure many things here
    def _hero_hp_score(self, state : GameState) -> float: ## A simple calculation of mean of hp ratios...

        heroes_mask = stateops.get_team_alive_mask(state)

        if not np.any(state.hp[heroes_mask] > 0):
            return 0.0

        return float(np.mean(state.hp[hereos_mask] / state.max_hp[heroes_mask]))

    def _boss_hp_score(self, state : GameState):

        return float(state.hp[self.boss_id] / state.max_hp[self.boss_is])

    def _hero_stamina_score(self, state : GameState):  # Stamina score for only the hereos

        heroes_mask = state.teams[Teams.HEROES]

        return float(np.mean(state.stamina[heroes_mask] / state.max_stamina[heroes_mask]))  ## There is no need for mask here, because we are calculating the stamina of all in potentials... But I guess we can separate the staminas out..
        

    def _boss_stamina_score(self, state : GameState):

        return float(state.stamina[self.boss_id] / state.max_stamina[self.boss_id])

    def _formation_score(self, state : GameState):  ## Measure how well the formation of hereos is 

        heroes_mask = stateops.get_team_alive_mask(state)
        active_positions = state.positions[heroes_mask]

        if len(active_positions) <= 1: # If there are 0 or 1 agents alive then there is no meaning to calculate the formation score
            return 0.0
        
        centroid = np.mean(active_positions, axis = 0)  # Does this mean that we are summing up all the pos (of shape [2,]) of all alive hero agents (axis = 0) and averaging over them ? Centroid  shape [2,].
        distances = np.sum(np.abs(active_positions - centroid), axis = 1)   # Axis 1 means that each position array subtracts and np.abs check if there are no negatives..And np.sum (of an array of size lets say (3,2) along  axis 1 would mean add each row, so end result is (3,1))
        
        return -float*(np.mean(distances))  ## This way we can see if the group is closer or not. If the negative is verylarge that means that the agents are spread out. This automatically affects the reward, so it tries to minimize that closer to 0, which would make the formation close..

    
    def _dealer_positions_score(self, state : GameState) -> float : # Reward if dealer is at a safe range than the boss and how much damage he did to the boss
        
        if state.hp[self.dealer_id] <= 0 or self.hp[self.boss_id] <= 0:
            return 0.0  #There is no meaning to calculate the reward of distance if both of them are dead.

        distance = stateops.get_distance(state, self.dealer_id, self.boss_is)
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

        if state.hp[boss_id] <= 0 or not np.any(state.hp[state.get_team_alive_mask(state, Teams.HEROES) > 0]):
            return 0.0 

        score = 0.0
        boss_to_tank = stateops.get_distance(state, self.tank_id, self.boss_id)

        for squishy_id in [self.healer_id, self.dealer_id]:
            boss_to_squishy = stateops.get_distance(state, squishy_id, self.boss_id)
            if boss_to_tank > boss_to_squishy:  # Boss should target weaker sections
                score += 3.0

            else:
                score -= 0.5

    def _get_handcrafted_potential(self, state : GameState) -> Tuple[float, float]:

        alive_hereos = len(stateops.get_alive_agent_ids(state, Teams.HEROES))

        hero_hp = self._hero_hp_score(state)
        boss_hp_penalty = -self._boss_hp_score(state)   ## a negative score for heroes as boss shouldnt be alive... 
        formation = self._formation_score(state)  # The amount of spread(is negative and agent minimizes that)...
        hero_stamina = self_hero_stamina_score(state)

        boss_hp = self._boss_hp_score(state)
        hero_hp_penalty = -hero_hp  # a penalty for boss as heroes are still alive
        boss_stamina = self._stamina_score(state)

        dealer_pos = self._dealer_position_score(state)
        healer_safe = self._healer_safety_score(state)
        tank_prot = self._tanker_protection_score(state)

        boss_squish = self._boss_attack_weak_score(state)

        ## Hero potential contains a lot of things summed weighted
        hero_pot = (hero_hp * 10.0) + (alive_heroes * 2.0) + (formation * 0.5) + \   
                   (hero_stamina * 0.5) + (boss_hp_penalty * 10.0) + \
                   dealer_pos + healer_safe + tank_prot

        boss_pot = (boss_hp * 10.0) + (hero_hp_penalty * 10.0) + (boss_stamina * 1.5) + boss_squish 

        return hero_pot, boss_pot