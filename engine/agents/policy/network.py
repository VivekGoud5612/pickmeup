import torch
import torch.nn as nn 
from torch.distributions import Categorical 
from engine.environment.state import GameState

def init_layer(layer : nn.Module, gain : float = 1.0, bias_constant : float = 0):
    
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain = gain)

    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_constant)
    
    return layer

class Actor(nn.Module):

    def __init__(self, raw_continuous_obs_size : int, action_space_size : int, role_embedding_dim : int = 32):  ## Raw Continuous Observations are of size 24 for each actor. Also role embeddings of size 32 is good (we would have 16 sized embeddings for each role for each network)
        super().__init__()

        self.role_embedding = nn.Embedding(num_embeddings = GameState.NUM_ROLES, embedding_dim = role_embedding_dim)  ## Here the number of embeddings is equal to the number of roles and each embedding dimension is 32
        self.entity_obs = nn.Linear(raw_continuous_obs_size, 128)   ## Map 24 sized obs to 128 semantic numbers to match embeddings. We RELU this to neatly map out activations
        
        relu_gain = nn.init.calculate_gain('relu')

        self.actor_body = nn.Sequential( ## We work directly with (num_envs, num_agents, obs, or role_embeds or others) - Shape is 128 + 32 = 160
            init_layer(nn.Linear(128 + role_embedding_dim * GameState.NUM_ROLES, 128), gain = relu_gain),  ##128 - 128 might overfit the data at the start..But now we have 256 - 128
            nn.ReLU(),
            init_layer(nn.Linear(128, 64), gain = relu_gain),
            nn.ReLU(),
        )

        self.actor_head = init_layer(nn.Linear(64, action_space_size), gain = 0.01) ## Small gain to initilize smaller weights at the start, alhough they are orhthogonal, creating near uniform action probabilites.
    
    def forward(self, observation : torch.Tensor, role_ids : torch.Tensor, action_mask : torch.Tensor | None = None) -> Categorical:

        local_features = torch.relu(self.entity_obs(observation))  # To a 128 dim blended tensor
        role_embeddings = self.role_embedding(role_ids) ## shape (num_envs, num_agents, 4, 32)

        shape = role_embeddings.shape 

        flat_embeds = torch.flatten(role_embeddings, start_dim = -2)  ## Flatten the last 2 layers so that he shape is (num_envs, num_agents, 4 * 32 = 128)

        fused_context = torch.cat([local_features, flat_embeds], dim = -1)  # concantenate.. so that the final shape (num_envs, num_agents, 160).

        features = self.actor_body(fused_context)  ## Shape (num_envs, num_agents, 64)
        logits = self.actor_head(features)  ## Shape (num_envs, num_agents, 8)..

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)  ## To set invalid actions to - infinity.. torch.finfo(type(logits - Mostly float32))

        return Categorical(logits = logits)  ## Outputs a distribution of actions



class Critic(nn.Module):

    def __init__(self, state_obs_size : int, role_embedding_dim : int = 64):   # Here the stateobs is of shape (batch, 96) and role_id_dim is (batch, 4)
        super().__init__()

        self.role_embeddings = nn.Embedding(num_embeddings = GameState.NUM_ROLES, embedding_dim = role_embedding_dim)
        self.state_encode = nn.Linear(state_obs_size, 256)  ## Map 96  values to 256 neurons to up the scale and store more.. but we need to see that they do not overfit

        tanh_gain = nn.init.calculate_gain('tanh')

        self.critic_body = nn.Sequential(  ## Again same as above and the shape is (num_envs, num_agents, 256 + 64 = 320)
            init_layer(nn.Linear(256 + role_embedding_dim * GameState.NUM_ROLES, 128), gain = tanh_gain),
            nn.Tanh(),
            init_layer(nn.Linear(128, 64), gain = tanh_gain),
            nn.Tanh(),
        )

        self.critic_head = init_layer(nn.Linear(64, 1), gain = 1.0)  #Scale of values should be high in the case of critic.

    def forward(self, global_state : torch.Tensor, global_role_ids : torch.Tensor):  ## Now these global role IDs are also the same as above

        global_features = torch.tanh(self.state_encode(global_state))  ## Just let the simple layer pass go through this activation so that the numbers do not explode
        global_role_embeds = self.role_embeddings(global_role_ids)  ## Shape (num_envs, num_agents, 64)

        shape = global_role_embeds.shape
        flat_embeds = torch.flatten(global_role_embeds, start_dim = -2) ## (num_envs, num_agents, 64 * 4 = 256)

        fused_global_context = torch.cat([global_features, flat_embeds], dim = -1) ## Maybe dim -1 means to act on the last dim.. 
        value_features = self.critic_body(fused_global_context) 
        value = self.critic_head(value_features)  ## Shape (num_envs, num_agents, 1)

        return value.squeeze(-1)  ## Shape (num-envs, num_agents,)
