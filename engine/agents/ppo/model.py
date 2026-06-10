import torch 
import torch.nn as nn
from torch.distributions import Categorical


def init_layer(layer : nn.Module, gain : float = 1.0, bias_const : float = 0.0) -> nn.Module:
    # Applies Orthogonal Initialization to Pytorch Linear layer
    
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain = gain)

        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)

    return layer


class MAPPO_ACTOR(nn.Module):
    # Dynamically instatiated, Decentralized actor for each agent.
    def __init__(self, local_obs_dim : int, action_dim : int, hidden_dim : int = 64):

        super().__init__()

        relu_gain = nn.init.calculate_gain('relu')

        self.actor_backbone = nn.Sequential(
            init_layer(nn.Linear(local_obs_dim, hidden_dim), gain = relu_gain),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, hidden_dim), gain = relu_gain),
            nn.ReLU(),
        )

        # Uses a microscopic gain(0.01) so early actions are unbiased,forcing maximum exploration at the start of training
        self.actor_head = init_layer(nn.Linear(hidden_dim, action_dim), gain = 0.01)

    
    def forward(self, local_obs : torch.Tensor, mask : torch.Tensor) -> Categorical:

        features = self.actor_backbone(local_obs)
        logits = self.actor_head(features)

        # Converts all 0's in action mask to massive negative value -1e9
        # When passed through the final softmax layer, this value effectively becomes probabilty=0.0
        masked_logits = logits.masked_fill(mask == 0, -1e9)

        return Categorical(logits = masked_logits)
    

class MAPPO_CRITIC(nn.Module):
    # Centralized value network ,that evaluates the global_state
    def __init__(self, global_obs_dim : int, hidden_dim : int):
        super().__init__()

        relu_gain = nn.init.calculate_gain('relu')

        self.value_backbone = nn.Sequential(
            init_layer(nn.Linear(global_obs_dim, hidden_dim), gain = relu_gain),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, hidden_dim), gain = relu_gain),
            nn.ReLU(),
        )

        # Uses standard gain(1.0) to quickly adapt baseline tracking steps,to fully predict using the range of normalised rewards
        self.value_head = init_layer(nn.Linear(hidden_dim, 1), gain = 1.0)


    def forward(self, global_state : torch.Tensor) -> torch.Tensor:

        features = self.value_backbone(global_state)
        value = self.value_head(features)

        return value
