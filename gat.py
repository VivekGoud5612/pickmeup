import numpy as np
from typing import Tuple, List
from engine.environment.state import GameState
from engine.agents.agent_data import AgentRole, Teams
from engine.environment.state_ops import StateOperations as stateops

class ObservationBuilder:
    VISION_RANGE = 3.0
    SELF_FEATURE_DIM = 9      # hp, stamina, norm_x, norm_y, block, invincible, 3x skills
    ENTITY_FEATURE_DIM = 5    # rel_x, rel_y, hp, stamina, alive
    TOTAL_CONTINUOUS_DIM = 24 # 9 + (3 entities * 5 features)

    @staticmethod
    def _extract_self_features(state: GameState, agent_id: int) -> np.ndarray:
        """Extracts the 9 core continuous/binary features for the self slot."""
        features = np.zeros(ObservationBuilder.SELF_FEATURE_DIM, dtype=np.float32)
        
        features[0] = state.hp[agent_id] / state.max_hp[agent_id]
        features[1] = state.stamina[agent_id] / state.max_stamina[agent_id]
        features[2] = state.positions[agent_id, 0] / state.grid_size
        features[3] = state.positions[agent_id, 1] / state.grid_size
        features[4] = 1.0 if state.is_blocking[agent_id] else 0.0
        features[5] = 1.0 if state.is_invincible[agent_id] else 0.0
        features[6] = 1.0 if state.cooldowns[agent_id, 0] == 0 else 0.0 # Basic ready
        features[7] = 1.0 if state.cooldowns[agent_id, 1] == 0 else 0.0 # Utility ready
        features[8] = 1.0 if state.cooldowns[agent_id, 2] == 0 else 0.0 # Ultimate ready
        
        return features

    @staticmethod
    def _extract_entity_features(state: GameState, self_id: int, target_id: int, check_vision: bool) -> Tuple[np.ndarray, int]:
        """
        Extracts the 5 relative features for an external entity slot.
        Applies shared fog-of-war vision masking natively.
        """
        features = np.zeros(ObservationBuilder.ENTITY_FEATURE_DIM, dtype=np.float32)
        role_id = int(state.roles[target_id])

        # If the target is dead, return default masked features
        if state.hp[target_id] <= 0:
            return features, role_id

        # Vision Verification Loop
        visible = True
        if check_vision:
            # Shared Vision: Visible if any living teammate is close enough to the target
            visible = False
            for teammate_id in range(3):
                if state.hp[teammate_id] > 0:
                    dist = stateops.get_distance(state, teammate_id, target_id)
                    if dist <= ObservationBuilder.VISION_RANGE:
                        visible = True
                        break

        if not visible:
            # Out of sight: returns placeholder metrics, but preserves the Role ID
            features[0:2] = -1.0 # Obscured position coordinates
            return features, role_id

        # Calculate exact relative offset coordinates requested in your footprint
        features[0] = (state.positions[target_id, 0] - state.positions[self_id, 0]) / state.grid_size
        features[1] = (state.positions[target_id, 1] - state.positions[self_id, 1]) / state.grid_size
        features[2] = state.hp[target_id] / state.max_hp[target_id]
        features[3] = state.stamina[target_id] / state.max_stamina[target_id]
        features[4] = 1.0 # Alive flag confirmation

        return features, role_id

    @staticmethod
    def build_partial_obs(state: GameState, agent_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the precise 24-dimension continuous vector and 4-dimension 
        role ID array for an agent's Actor Network.
        """
        continuous_vector = np.zeros(ObservationBuilder.TOTAL_CONTINUOUS_DIM, dtype=np.float32)
        role_ids = np.zeros(4, dtype=np.int32)
        
        role = state.roles[agent_id]
        is_boss = (role == AgentRole.BOSS)
        boss_id = 3

        # 1. Populate Slot 0: Self (0 - 8)
        continuous_vector[0:9] = ObservationBuilder._extract_self_features(state, agent_id)
        role_ids[0] = int(role)

        # 2. Determine Deterministic Slot Order for the remaining 3 entities
        if not is_boss:
            # Hero Perspective layout tracking
            if role == AgentRole.TANK:
                external_entities = [int(AgentRole.DEALER), int(AgentRole.HEALER), boss_id]
            elif role == AgentRole.DEALER:
                external_entities = [int(AgentRole.TANK), int(AgentRole.HEALER), boss_id]
            else: # Healer
                external_entities = [int(AgentRole.TANK), int(AgentRole.DEALER), boss_id]
        else:
            # Boss Perspective layout tracking (sees all 3 heroes as enemies)
            external_entities = [int(AgentRole.TANK), int(AgentRole.DEALER), int(AgentRole.HEALER)]

        # 3. Harvest and pack external slots sequentially
        for slot_idx, target_id in enumerate(external_entities):
            # Only apply fog-of-war vision masking if a Hero is trying to look at the Boss
            # (Heroes always know where teammates are; Boss always tracks intruders)
            check_vision = (not is_boss and target_id == boss_id)
            
            feats, target_role = ObservationBuilder._extract_entity_features(
                state, agent_id, target_id, check_vision=check_vision
            )
            
            # Slide features into their precise 5-element stride mapping
            start_stride = 9 + (slot_idx * ObservationBuilder.ENTITY_FEATURE_DIM)
            end_stride = start_stride + ObservationBuilder.ENTITY_FEATURE_DIM
            continuous_vector[start_stride:end_stride] = feats
            
            # Track parallel discrete categorical array indexes
            role_ids[slot_idx + 1] = target_role

        return continuous_vector, role_ids

    @staticmethod
    def build_full_state(state: GameState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compiles the global omniscient tracking vector for the Centralized Critic.
        Concatenates all 4 agents' partial observation profiles sequentially.
        """
        num_agents = state.num_agents
        global_features = np.zeros(num_agents * ObservationBuilder.TOTAL_CONTINUOUS_DIM, dtype=np.float32)
        global_roles = np.zeros(num_agents * 4, dtype=np.int32)

        for idx in range(num_agents):
            feat_start = idx * ObservationBuilder.TOTAL_CONTINUOUS_DIM
            feat_end = feat_start + ObservationBuilder.TOTAL_CONTINUOUS_DIM
            
            role_start = idx * 4
            role_end = role_start + 4
            
            feats, roles = ObservationBuilder.build_partial_obs(state, idx)
            
            global_features[feat_start:feat_end] = feats
            global_roles[role_start:role_end] = roles

        return global_features, global_roles