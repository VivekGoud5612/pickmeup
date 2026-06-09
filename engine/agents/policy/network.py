import torch
import torch.nn as nn 
from torch.distributions import Categorical 

NUM_ROLES = 4

def init_layer(layer : nn.Module, gain : float = 1.0, bias_constant : float = 0):
    
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain = gain)

    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_constant)
    
    return layer

class Actor(nn.Module):

    def __init__(self, raw_continuous_obs_size : int, action_space_size : int, role_embedding_dim : int = 16):  ## Raw Continuous Observations are of size 24 for each actor. Also role embeddings of size 16 is good (we would have 16 sized embeddings for each role for each network)
        super().__init__()

        self.role_embedding = nn.Embedding(num_embeddings = NUM_ROLES, embedding_dim = role_embedding_dim)  ## Here the number of embeddings is equal to the number of roles and each embedding dimension is 16
        total_local_role_embeds = NUM_ROLES * role_embedding_dim  # This is so that when we convert (batch, 4, 16) -> (batch, 64).. we need the shape

        self.entity_obs = nn.Linear(raw_continuous_obs_size, 64)   ## Map 24 sized obs to 64 semantic numbers to match embeddings. We RELU this to neatly map out activations

        fused_input_dim = 64 + total_local_role_embeds  ## The first 64 is from output of self.entity_obs.. this is where we combine both
        
        relu_gain = nn.init.calculate_gain('relu')

        self.actor_body = nn.Sequential(
            init_layer(nn.Linear(fused_input_dim, 64), gain = relu_gain),  ##128 - 128 might overfit the data at the start..
            nn.ReLU(),
            init_layer(nn.Linear(64, 64), gain = relu_gain),
            nn.ReLU(),
        )

        self.actor_head = init_layer(nn.Linear(64, action_space_size), gain = 0.01) ## Small gain to initilize smaller weights at the start, alhough they are orhthogonal, creating near uniform action probabilites.
    
    def forward(self, observation : torch.Tensor, role_ids : torch.Tensor, action_mask : torch.Tensor | None = None) -> Categorical:

        assert observation.ndim == 2   # Of shape (batch_size, 24) 
        assert role_ids.ndim == 2 ## shape (batch_size, 4)

        local_features = torch.relu(self.entity_obs(observation))  # To a 64 dim blended tensor

        role_embeddings = self.role_embedding(role_ids) 
        flattened_embeddings = role_embeddings.view(role_embeddings.size(0), -1)  ## We view the same where the first dim is as is and the last are combined. So (batch_size, 64)...No that whole thing means do not touch the very fist dim, and combine every other and put that in the last dim (as we only have 2 dims , it goes to dim 1)

        fused_context = torch.cat([local_features, flattened_embeddings], dim = -1)  # concantenate.. so that the final shape (batch, 128).

        features = self.actor_body(fused_context)
        logits = self.actor_head(features)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

        return Categorical(logits = logits)



class Critic(nn.Module):

    def __init__(self, state_obs_size : int, role_embedding_dim : int = 16):   # Here the stateobs is of shape (batch, 96) and role_embedding_dim is (batch, 4)
        super().__init__()

        self.role_embeddings = nn.Embedding(num_embeddings = NUM_ROLES, embedding_dim = role_embedding_dim)
        total_global_role_embeds = NUM_ROLES * role_embedding_dim  ## OF size (64)

        self.state_encode = nn.Linear(state_obs_size, 128)

        fused_global_size = 128 + total_global_role_embeds  #128 from state encoded size, so total size is 192

        tanh_gain = nn.init.calculate_gain('tanh')

        self.critic_body = nn.Sequential(
            init_layer(nn.Linear(fused_global_size, 128), gain = tanh_gain),
            nn.Tanh(),
            init_layer(nn.Linear(128, 64), gain = tanh_gain),
            nn.Tanh(),
        )

        self.critic_head = init_layer(nn.Linear(64, 1), gain = 1.0)  #Scale of values should be high in the case of critic.

    def forward(self, global_state : torch.Tensor, global_role_ids : torch.Tensor):

        assert global_state.ndim == 2  # Shape (batch_size, 96)
        assert global_role_ids.ndim == 2 #Shape (batch_size, 4)

        global_features = torch.tanh(self.state_encode(global_state))

        global_role_embed = self.role_embeddings(global_role_ids)
        flattened_embeddings_global = global_role_embed.view(global_role_embed.size(0), -1)  # Same as actor, but I still donot know about this view method. Why do we only act on last dim but not on last dims apart from the very first..No that whole thing means do not touch the very fist dim, and combine every other and put that in the last dim (as we only have 2 dims , it goes to dim 1)

        fused_global_context = torch.cat([global_features, flattened_embeddings_global], dim = -1) ## Maybe dim -1 means to act on the last dim.. 
        value_features = self.critic_body(fused_global_context) 
        value = self.critic_head(value_features)

        return value
