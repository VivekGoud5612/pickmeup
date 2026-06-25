import torch 
import torch.nn as nn

class Actor_Critic(nn.Module):
    def __init__(self,state_size,action_size):
        super(Actor_Critic,self).__init__()

        self.actor=nn.Sequential(
            nn.Linear(state_size,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,action_size),
        )

        self.critic=nn.Sequential(
            nn.Linear(state_size,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,1),
        )

    def forward(self,state:torch.Tensor,mask:torch.Tensor):

        logits=self.actor(state)
        value=self.critic(state)

        if mask is not None:
            logits=logits.masked_fill(~mask,float('-inf'))

        return logits,value