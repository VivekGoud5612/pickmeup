import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

# Import your core engine components
from engine.environment.state import Gastateops
from engine.environment.action_sequencer import ActionSequencermeState
from engine.environment.state_ops import StateOperations as 
from engine.agents.policy.reward_calculator import RewardCalculator
from engine.agents.policy.observation_builder import ObservationBuilder

class RaidEnv(gym.Env):
    """
    Custom Multi-Agent Gymnasium Environment for a 4-Agent Boss Raid.
    Outputs Local Obs for the Actor and Global State for the MAPPO Critic.
    """
    def __init__(self, grid_size: int = 20, max_steps: int = 500):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_count = 0
        self.num_agents = 4

        # Instantiate backend DoD components
        self.state = GameState(grid_size=self.grid_size)
        self.reward_calc = RewardCalculator()
        self.obs_builder = ObservationBuilder(grid_size=self.grid_size)

        # =====================================================================
        # GYMNASIUM SPACES DEFINITION
        # =====================================================================
        # Action Space: 4 agents, each choosing from 8 discrete actions
        self.action_space = spaces.MultiDiscrete([8] * self.num_agents)

        # Observation Space: Dict containing local obs, roles, and the MAPPO global state
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents, 24), dtype=np.float32),
            "roles": spaces.Box(low=0, high=3, shape=(self.num_agents, 4), dtype=np.int32),
            "global_state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents, 24 * self.num_agents), dtype=np.float32)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)         
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
            "active_masks": self._get_active_masks()
        }

        return obs_dict, info

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # 1. Snapshot State & Extract Historical Values for Reward Shaping (PBRS)
        old_state_snapshot = self.state.clone_snapshot()
        old_values = self.state.current_step_value_predictions 
        
        # 2. Clear Transient Trackers
        self.state.exploration_bonus_triggered.fill(0.0)

        # 3. Pre-process actions against strict mechanics masks
        processed_mask = np.ones(self.num_agents, dtype=bool)
        current_masks = self._get_action_masks()
        for a_idx in range(self.num_agents):
            if not current_masks[a_idx, actions[a_idx]]:
                processed_mask[a_idx] = False

        # 4. Resolve Physics & Mechanics
        ActionSequencer.resolve_step(self.state, actions)

        # 5. Calculate Decomposed Rewards
        new_values = self.state.current_step_value_predictions
        reward_breakdown = self.reward_calc.calculate_decomposed_reward(
            old_state=old_state_snapshot,
            new_state=self.state,
            processed_mask=processed_mask,
            phase=1,
            old_values=old_values,
            new_values=new_values
        )
        rewards = reward_breakdown['total_rewards']

        # 6. Evaluate Terminations & Truncations
        is_terminal = stateops.is_terminal(self.state)
        truncated = bool(self.step_count >= self.max_steps)

        terminated_array = np.zeros(self.num_agents, dtype=np.float32)
        for a_idx in range(self.num_agents):
            was_alive = old_state_snapshot.hp[a_idx] > 0
            is_alive = self.state.hp[a_idx] > 0
            if (was_alive and not is_alive) or is_terminal:
                terminated_array[a_idx] = 1.0

        # 7. Package Outputs
        next_obs_dict = self._get_observations()
        
        info = {
            "action_masks": self._get_action_masks(),
            "active_masks": self._get_active_masks(),
            "reward_breakdown": reward_breakdown, 
        }

        # Terminal Observation Trap for GAE Bootstrapping
        if is_terminal or truncated:
            info["terminal_observation"] = next_obs_dict

        return next_obs_dict, rewards, terminated_array, truncated, info

    # =====================================================================
    # INTERNAL HELPERS
    # =====================================================================
    def _get_observations(self) -> Dict[str, np.ndarray]:
        env_obs = np.zeros((self.num_agents, 24), dtype=np.float32)
        env_roles = np.zeros((self.num_agents, 4), dtype=np.int32)
        
        for a_idx in range(self.num_agents):
            obs, roles = self.obs_builder.build_partial_obs(self.state, a_idx)
            env_obs[a_idx] = obs
            env_roles[a_idx] = roles
            
        # MAPPO Global State Construction: Flatten all local obs, copy it for all agents
        flattened_global = env_obs.flatten() # Shape: (96,)
        global_state = np.tile(flattened_global, (self.num_agents, 1)) # Shape: (4, 96)
            
        return {"obs": env_obs, "roles": env_roles, "global_state": global_state}

    def _get_action_masks(self) -> np.ndarray:
        masks = np.zeros((self.num_agents, 8), dtype=bool)
        for a_idx in range(self.num_agents):
            masks[a_idx] = stateops.get_action_mask(self.state, a_idx)
        return masks

    def _get_active_masks(self) -> np.ndarray:
        active = np.zeros(self.num_agents, dtype=np.float32)
        for a_idx in range(self.num_agents):
            if self.state.hp[a_idx] > 0:
                active[a_idx] = 1.0
        return active