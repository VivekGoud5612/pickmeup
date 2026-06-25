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

    def __init__(self, raw_continuous_obs_size : int, action_space_size : int = 8):  ## Raw Continuous Observations are of size 24 for each actor. Also role embeddings of size 32 is good (we would have 16 sized embeddings for each role for each network)
        super().__init__()
        
        relu_gain = nn.init.calculate_gain('relu')

        self.actor_body = nn.Sequential( ## We work directly with (num_envs, num_agents, obs, or role_embeds or others) - Shape is 128 + 32 = 160
            init_layer(nn.Linear(raw_continuous_obs_size, 128), gain = relu_gain),  ##128 - 128 might overfit the data at the start..But now we have 256 - 128
            nn.ReLU(),
            init_layer(nn.Linear(128, 64), gain = relu_gain),
            nn.ReLU(),
        )

        self.actor_head = init_layer(nn.Linear(64, action_space_size), gain = 0.01) ## Small gain to initilize smaller weights at the start, alhough they are orhthogonal, creating near uniform action probabilites.
    
    def forward(self, observation : torch.Tensor, action_mask : torch.Tensor | None = None) -> Categorical:

        features = self.actor_body(observation)  ## Shape (num_envs, num_agents, 24)
        logits = self.actor_head(features)  ## Shape (num_envs, num_agents, 8)..

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)  ## To set invalid actions to - infinity.. torch.finfo(type(logits - Mostly float32))

        return Categorical(logits = logits)  ## Outputs a distribution of actions



class Critic(nn.Module):

    def __init__(self, state_obs_size : int):   # Here the stateobs is of shape (batch, 96) and role_id_dim is (batch, 4)
        super().__init__()

        tanh_gain = nn.init.calculate_gain('tanh')

        self.critic_body = nn.Sequential(  ## Again same as above and the shape is (num_envs, num_agents, 256 + 64 = 320)
            init_layer(nn.Linear(state_obs_size, 128), gain = tanh_gain),
            nn.Tanh(),
            init_layer(nn.Linear(128, 64), gain = tanh_gain),
            nn.Tanh(),
        )

        self.critic_head = init_layer(nn.Linear(64, 1), gain = 1.0)  #Scale of values should be high in the case of critic.

    def forward(self, global_state : torch.Tensor):  ## Now these global role IDs are also the same as above

        value_features = self.critic_body(global_state) 
        value = self.critic_head(value_features)  ## Shape (num_envs, num_agents, 1)

        return value.squeeze(-1)  ## Shape (num-envs, num_agents,)
