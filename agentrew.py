import numpy as np
from typing import Dict, Tuple, Any
from engine.actions.action_handler import ActionTypes, AgentRole, Team
from engine.environment.state_ops import StateOperations as stateops
from engine.environment.state import GameState

class RewardCalculator:
    def __init__(self, gamma: float = 0.80):
        self.gamma = gamma

        # Base Environment Penalties/Bounties
        self.time_step_penalty = -0.1
        self.invalid_action_penalty = -0.1
        self.win_bounty = 100.0
        self.death_penalty = -50.0

        # Hardcoded IDs matching your team structure
        self.tank_id = 0
        self.dealer_id = 1
        self.healer_id = 2
        self.boss_id = 3

    # =====================================================================
    # HANDCRAFTED SUB-SCORES
    # =====================================================================
    def _hero_hp_score(self, state: GameState) -> float:
        heroes_mask = state.team_masks[Team.HEROES]
        if not np.any(state.hp[heroes_mask] > 0):
            return 0.0
        return float(np.mean(state.hp[heroes_mask] / state.max_hp[heroes_mask]))

    def _boss_hp_score(self, state: GameState) -> float:
        return float(state.hp[self.boss_id] / state.max_hp[self.boss_id])

    def _stamina_score(self, state: GameState) -> float:
        return float(np.mean(state.stamina / state.max_stamina))

    def _formation_score(self, state: GameState) -> float:
        """Measures how tightly grouped the living heroes are."""
        heroes_mask = state.team_masks[Team.HEROES] & (state.hp > 0)
        active_positions = state.positions[heroes_mask]
        if len(active_positions) <= 1:
            return 0.0
        
        centroid = np.mean(active_positions, axis=0)
        distances = np.sum(np.abs(active_positions - centroid), axis=1)
        return -float(np.mean(distances)) # Closer group = closer to 0 (higher potential)

    def _dealer_position_score(self, state: GameState) -> float:
        """Dealer wants to be exactly 2-3 tiles away from the Boss."""
        if state.hp[self.dealer_id] <= 0 or state.hp[self.boss_id] <= 0:
            return 0.0
        dist = stateops.get_distance(state, self.dealer_id, self.boss_id)
        return 1.0 if (2 <= dist <= 3) else -float(abs(dist - 2.5))

    def _healer_safety_score(self, state: GameState) -> float:
        """Healer wants distance from the Boss, proximity to the Tank."""
        if state.hp[self.healer_id] <= 0:
            return 0.0
        score = 0.0
        if state.hp[self.boss_id] > 0:
            score += stateops.get_distance(state, self.healer_id, self.boss_id) * 0.5
        if state.hp[self.tank_id] > 0:
            score -= stateops.get_distance(state, self.healer_id, self.tank_id) * 0.5
        return float(score)

    def _tank_protection_score(self, state: GameState) -> float:
        """Tank gets points for standing between the Boss and the squishies."""
        if state.hp[self.tank_id] <= 0 or state.hp[self.boss_id] <= 0:
            return 0.0
        
        score = 0.0
        tank_to_boss = stateops.get_distance(state, self.tank_id, self.boss_id)
        
        for squishy_id in [self.dealer_id, self.healer_id]:
            if state.hp[squishy_id] > 0:
                squishy_to_boss = stateops.get_distance(state, squishy_id, self.boss_id)
                if tank_to_boss < squishy_to_boss:
                    score += 2.0
        return float(score)

    def _get_handcrafted_potential(self, state: GameState) -> Tuple[float, float]:
        """Compiles sub-scores into separate potentials for both teams."""
        alive_heroes = int(np.sum((state.team_masks[Team.HEROES]) & (state.hp > 0)))
        
        # Base structural values
        hero_hp = self._hero_hp_score(state)
        boss_hp_penalty = -self._boss_hp_score(state)
        formation = self._formation_score(state)
        stamina = self._stamina_score(state)

        # Tactical positions
        dealer_pos = self._dealer_position_score(state)
        healer_safe = self._healer_safety_score(state)
        tank_prot = self._tank_protection_score(state)

        # Combine matching your blueprint
        hero_pot = (hero_hp * 10.0) + (alive_heroes * 2.0) + (formation * 0.5) + \
                   (stamina * 0.5) + (boss_hp_penalty * 10.0) + \
                   dealer_pos + healer_safe + tank_prot

        # Boss potential is inverse to hero performance
        boss_pot = (self._boss_hp_score(state) * 10.0) - (hero_hp * 10.0)

        return hero_pot, boss_pot

    # =====================================================================
    # MASTER CALCULATOR WITH PHASE ANNEALING
    # =====================================================================
    def calculate_rewards(self, 
                          old_state: GameState, 
                          new_state: GameState, 
                          processed_mask: np.ndarray,
                          phase: int, 
                          old_values: np.ndarray, 
                          new_values: np.ndarray) -> np.ndarray:
        """
        Calculates rewards for all 4 agents based on your 4-Phase system schedule.
        - processed_mask: boolean array showing if the agent executed a valid action.
        - old_values / new_values: Critic predictions for each agent [shape: (4,)]
        """
        num_agents = new_state.num_agents
        rewards = np.zeros(num_agents, dtype=np.float32)

        # 1. Determine Alpha based on your Phase parameters
        if phase == 1:   alpha = 0.0
        elif phase == 2: alpha = 0.2
        elif phase == 3: alpha = 0.5
        else:            alpha = 1.0 # Phase 4: Pure Critic Value

        # 2. Get Handcrafted Potentials
        old_hero_hand, old_boss_hand = self._get_handcrafted_potential(old_state)
        new_hero_hand, new_boss_hand = self._get_handcrafted_potential(new_state)

        # 3. Blend Potentials per Agent
        # Each agent evaluates their team's perspective combined with their personal critic value
        for idx in range(num_agents):
            is_boss = (new_state.roles[idx] == AgentRole.BOSS)
            
            # Extract relevant handcrafted block
            old_hand = old_boss_hand if is_boss else old_hero_hand
            new_hand = new_boss_hand if is_boss else new_hero_hand

            # Apply your blending formula
            old_phi = ((1.0 - alpha) * old_hand) + (alpha * old_values[idx])
            new_phi = ((1.0 - alpha) * new_hand) + (alpha * new_values[idx])

            # Shaped PBRS reward calculation
            rewards[idx] += (self.gamma * new_phi) - old_phi

        # 4. Step Penalties & Invalid Move Interceptions
        for idx in range(num_agents):
            if new_state.hp[idx] > 0:
                rewards[idx] += self.time_step_penalty
                if not processed_mask[idx]: # If action sequencer flagged them as invalid
                    rewards[idx] += self.invalid_action_penalty

        # 5. Terminal Breakdowns (Win Bounties & Death Penalties)
        heroes_mask = new_state.team_masks[Team.HEROES]
        boss_id = 3

        for idx in range(num_agents):
            was_alive = old_state.hp[idx] > 0
            is_alive = new_state.hp[idx] > 0

            if was_alive and not is_alive:
                rewards[idx] += self.death_penalty
                
                if idx == boss_id:
                    # Boss died! Living heroes claim the match bounty
                    for hero_id in np.where(heroes_mask & (new_state.hp > 0))[0]:
                        rewards[hero_id] += self.win_bounty
                else:
                    # Hero died! Give the Boss their bounty
                    rewards[boss_id] -= self.death_penalty

        return rewards