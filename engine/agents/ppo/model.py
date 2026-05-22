import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    
    def __init__(self, state_size : int, actor_size : int):
        super(ActorCritic, self).__init__()
    
        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128,64),
            nnn.Tanh(),
            nn.Linear(64, action_size),
        )
    
        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, state : torch.Tensor, mask : torch.Tensor):
        logits = self.actor(state)
        value = self.critic(state)
    
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        
        return logits, value