import torch
import torch.nn as nn 
from torch.distributions import Categorical 


def init_layer(layer : nn.Module, gain : float = 1.0, bias_constant : float = 0):
    
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain = gain)

    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_costant)
    
    return layer

class Actor(nn.Module):

    def __init__(self, obs_size, action_space_size):
        super().__init__()


        relu_gain = nn.init.calculate_gain('relu')

        self.actor_body = nn.Sequential(
            init_layer(nn.Linear(obs_size, 64), gain = relu_gain),  ##128 - 128 might overfit the data at the start... As our observation vector is of size 128
            nn.ReLU(),
            init_layer(nn.Linear(64, 64), gain = relu_gain),
            nn.ReLU(),
        )

        self.actor_head = init_layer(nn.Linear(64, action_space_size), gain = 0.01) ## Small gain to initilize smaller weights at the start, alhough they are orhthogonal, creating near uniform action probabilites.
    
    def forward(self, observation : torch.Tensor, action_mask : torch.Tensor | None = None) -> Categorical:

        assert observation.ndim == 2

        features = self.actor_body(observation)
        logits = self.actor_head(features)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

        return Categorical(logits = logits)



class Critic(nn.Module):

    def __init__(self, state_size : int):
        super().__init__()

        tanh_gain = nn.init.calculate_gain('tanh')

        self.critic_body = nn.Sequential(
            init_layer(nn.Linear(state_size, 128), gain = tanh_gain),
            nn.Tanh(),
            init_layer(nn.Linear(128, 64), gain = tanh_gain),
            nn.Tanh(),
        )

        self.critic_head = init_layer(nn.Linear(64, 1), gain = 1.0)  #Scale of values should be high in the case of critic.

    def forward(self, global_state : torch.Tensor):

        assert global_state.ndim = 2

        value_features = self.critic_body(global_state) 
        value = self.critic_head(value_features)

        return value
