import numpy as np
from typing import Tuple, Dict, Any

# Import your core engine components
from engine.environment.state import GameState
from engine.environment.state_ops import StateOperations as stateops
from engine.environment.action_sequencer import ActionSequencer
from engine.agents.policy.reward_calculator import RewardCalculator
from engine.agents.policy.observation_builder import ObservationBuilder

class RaidEnv:
    def __init__(self, grid_size: int = 20, max_steps: int = 200):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_agents = 4
        self.step_count = 0

        # Instantiate the stateless backend singletons
        self.state = GameState(grid_size=self.grid_size)
        self.reward_calc = RewardCalculator()
        self.obs_builder = ObservationBuilder(grid_size=self.grid_size)

        # Mapping dictionary based on your reference
        self.agents = {
            0: "Tank",
            1: "Dealer",
            2: "Healer",
            3: "Boss"
        }

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Resets the physical environment to its starting state.
        """
        self.step_count = 0
        self.state.reset()

        # Seed initial footprint into the exploration tracker
        for agent_idx in range(self.num_agents):
            x, y = self.state.positions[agent_idx]
            team = self.state.teams[agent_idx]
            self.state.team_visited_tiles[team, int(x), int(y)] = True

        # Generate initial frames
        obs_dict = self._get_observations()
        
        info = {
            "action_masks": self._get_action_masks(),
            "active_masks": self._get_active_masks(),
            "handcrafted_potential": self._get_handcrafted_potentials()
        }

        return obs_dict, info

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Executes a single frame of the simulation. No value functions used here.
        """
        self.step_count += 1
        
        # 1. Snapshot State for Reward Calculation
        old_state_snapshot = self.state.clone_snapshot()
        
        # 2. Clear Transient Trackers
        self.state.exploration_bonus_triggered.fill(0.0)

        # 3. Pre-process actions against strict mechanics masks
        processed_mask = np.ones(self.num_agents, dtype=bool)
        current_masks = self._get_action_masks()
        
        for a_idx in range(self.num_agents):
            chosen_action = actions[a_idx]
            if not current_masks[a_idx, chosen_action]:
                # Invalid action attempted
                processed_mask[a_idx] = False

        # 4. Resolve the Physics & Mechanics Engine
        ActionSequencer.resolve_step(self.state, actions)

        # 5. Calculate RAW Combat Rewards (No Neural Network Values)
        # Note: You'll need to ensure your RewardCalculator has a method that just gets the raw combat additions
        combat_rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        if old_state_snapshot.hp[1] > 0: # Dealer
            self.reward_calc._combat_reward_for_dealer(self.state, combat_rewards)
        if old_state_snapshot.hp[0] > 0: # Tank
            self.reward_calc._combat_reward_for_tank(self.state, combat_rewards)
        if old_state_snapshot.hp[2] > 0: # Healer
            self.reward_calc._combat_reward_for_healer(self.state, combat_rewards)
        if old_state_snapshot.hp[3] > 0: # Boss
            self.reward_calc._combat_reward_for_boss(self.state, combat_rewards)

        # 6. Evaluate Terminations (Win/Loss) & Truncations (Time Limit)
        is_terminal = stateops.is_terminal(self.state)
        truncated = bool(self.step_count >= self.max_steps)

        terminated_array = np.zeros(self.num_agents, dtype=np.float32)
        for a_idx in range(self.num_agents):
            was_alive = old_state_snapshot.hp[a_idx] > 0
            is_alive = self.state.hp[a_idx] > 0
            if (was_alive and not is_alive) or is_terminal:
                terminated_array[a_idx] = 1.0

        # Add win/death penalties to combat_rewards directly based on terminations
        for a_idx in range(self.num_agents):
            if terminated_array[a_idx] == 1.0 and is_terminal:
                if a_idx == self.reward_calc.boss_id:
                    combat_rewards[self.reward_calc.boss_id] -= self.reward_calc.death_penalty
                else:
                    combat_rewards[a_idx] += self.reward_calc.win_bounty
            elif terminated_array[a_idx] == 1.0:
                combat_rewards[a_idx] += self.reward_calc.death_penalty

        # 7. Package Outputs
        next_obs_dict = self._get_observations()

        info = {
            "action_masks": self._get_action_masks(),
            "active_masks": self._get_active_masks(),
            # Expose the pure heuristic potential for main.py to blend with Values
            "handcrafted_potential": self._get_handcrafted_potentials() 
        }

        return next_obs_dict, combat_rewards, terminated_array, truncated, info

    def close(self):
        pass

    # =====================================================================
    # INTERNAL HELPERS
    # =====================================================================
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Harvests complete environment vision arrays."""
        env_obs = np.zeros((self.num_agents, 24), dtype=np.float32)
        env_roles = np.zeros((self.num_agents, 4), dtype=np.int32)
        
        for a_idx in range(self.num_agents):
            obs, roles = self.obs_builder.build_partial_obs(self.state, a_idx)
            env_obs[a_idx] = obs
            env_roles[a_idx] = roles
            
        return {"obs": env_obs, "roles": env_roles}

    def _get_action_masks(self) -> np.ndarray:
        """Extracts strict binary rules for available moves from StateOperations."""
        action_masks = np.zeros((self.num_agents, 8), dtype=bool)
        for a_idx in range(self.num_agents):
            action_masks[a_idx] = stateops.get_action_mask(self.state, a_idx)
        return action_masks

    def _get_active_masks(self) -> np.ndarray:
        """Extracts active status (1.0 if alive, 0.0 if dead)."""
        active_masks = np.zeros(self.num_agents, dtype=np.float32)
        for a_idx in range(self.num_agents):
            if stateops.is_alive(self.state, a_idx):
                active_masks[a_idx] = 1.0
        return active_masks

    def _get_handcrafted_potentials(self) -> np.ndarray:
        """Calculates the heuristic state values (Φ_handcrafted)."""
        hero_pot, boss_pot = self.reward_calc._get_handcrafted_potential(self.state)
        potentials = np.zeros(self.num_agents, dtype=np.float32)
        
        for a_idx in range(self.num_agents):
            if self.state.roles[a_idx] == 3: # Boss
                potentials[a_idx] = boss_pot
            else:
                potentials[a_idx] = hero_pot
                
        return potentials