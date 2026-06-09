import torch
import torch.nn as nn
import numpy as np
from engine.environment.state import GameState
from engine.agents.agent_data import AgentRole, Teams
from engine.environment.state_ops import StateOperations as stateops

class MultiAgentObservationEncoder(nn.Module):
    def __init__(self, raw_feature_dim: int = 10, role_embedding_dim: int = 16, output_dim: int = 128):
        super().__init__()
        
        # 1. Learnable Role Embeddings: 4 Roles (Tank=0, Dealer=1, Healer=2, Boss=3)
        self.role_embedding = nn.Embedding(num_embeddings=4, embedding_dim=role_embedding_dim)
        
        # 2. Continuous Feature Projection Layer
        self.feature_projection = nn.Linear(raw_feature_dim, 48)
        
        # 3. Combined Fusion Network (Projects to your target 128-dim tensor)
        self.fusion_network = nn.Sequential(
            nn.Linear(48 + role_embedding_dim, 96),
            nn.ReLU(),
            nn.Linear(96, output_dim),
            nn.LayerNorm(output_dim) # Stabilizes training across different agent roles
        )

    def extract_raw_features(self, state: GameState, agent_id: int, boss_id: int = 3) -> np.ndarray:
        """
        Slices the global continuous game state matrices from the viewpoint of a single agent.
        Output Shape: (10,) continuous features.
        """
        features = []
        
        # Self Metrics (Normalized)
        features.append(state.hp[agent_id] / state.max_hp[agent_id])
        features.append(state.stamina[agent_id] / state.max_stamina[agent_id])
        features.append(state.positions[agent_id, 0] / state.grid_size) # Normalized X
        features.append(state.positions[agent_id, 1] / state.grid_size) # Normalized Y
        
        # Action Flags
        features.append(1.0 if state.is_blocking[agent_id] else 0.0)
        features.append(1.0 if state.is_invincible[agent_id] else 0.0)
        
        # Spatial Metrics Relative to the Threat (The Boss)
        if agent_id == boss_id:
            # If I am the boss, find the distance to the closest living hero
            heroes_mask = state.team_masks[Teams.HEROES] & (state.hp > 0)
            if np.any(heroes_mask):
                hero_ids = np.where(heroes_mask)[0]
                dists = [stateops.get_distance(state, agent_id, h_id) for h_id in hero_ids]
                features.append(min(dists) / state.grid_size)
            else:
                features.append(0.0)
            features.extend([0.0, 0.0, 0.0]) # Padding metrics to keep shapes symmetric
        else:
            # If I am a hero, track my direct spatial mapping to the Boss
            dist_to_boss = stateops.get_distance(state, agent_id, boss_id)
            features.append(dist_to_boss / state.grid_size)
            features.append((state.positions[boss_id, 0] - state.positions[agent_id, 0]) / state.grid_size) # Relative X
            features.append((state.positions[boss_id, 1] - state.positions[agent_id, 1]) / state.grid_size) # Relative Y
            features.append(state.hp[boss_id] / state.max_hp[boss_id]) # Boss Health threat assessment

        return np.array(features, dtype=np.float32)

    def forward(self, state: GameState) -> torch.Tensor:
        """
        Processes the global GameState and outputs a dense embedding tensor for all agents.
        Output Shape: (num_agents, 128)
        """
        num_agents = state.num_agents
        
        # 1. Compile raw numpy slices and roles from the state arrays
        raw_features_list = []
        roles_list = []
        
        for idx in range(num_agents):
            raw_features_list.append(self.extract_raw_features(state, idx))
            roles_list.append(int(state.roles[idx])) # Direct IntEnum integer extraction

        # 2. Convert batch targets into PyTorch tensors safely
        raw_features_tensor = torch.tensor(np.array(raw_features_list), dtype=torch.float32)
        roles_tensor = torch.tensor(roles_list, dtype=torch.long)

        # 3. Process categorical role codes through learnable embedding space
        role_embeds = self.role_embedding(roles_tensor) # Shape: (num_agents, 16)
        
        # 4. Project continuous spatial/vital statistics
        projected_features = torch.relu(self.feature_projection(raw_features_tensor)) # Shape: (num_agents, 48)
        
        # 5. Concatenate streams and fuse down to your target 128 dimension block
        combined = torch.cat([projected_features, role_embeds], dim=-1) # Shape: (num_agents, 64)
        dense_embeddings = self.fusion_network(combined) # Shape: (num_agents, 128)
        
        return dense_embeddings