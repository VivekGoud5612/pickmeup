import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List
from model import Actor_Critic

lr=3e-4
gamma=0.99
lam=0.95
c1=0.5
eps_clip=0.2
K_epoch=4

class Agent:
    def __init__(self,state_size:int,action_size:int,agent_id:int,role:str):
        self.state_size=state_size
        self.action_size=action_size

        self.agent_id=agent_id
        self.role=role
        self.c2=0.1

        self.policy=Actor_Critic(state_size,action_size)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=lr)

        self.memory={
            "states":[],
            "actions":[],
            "log_probs":[],
            "values":[],
            "rewards":[],
            "masks":[],
            "dones":[],
        }

    def store_reward(self,reward:float,done:bool):
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)

    def get_action(self,observation:np.ndarray ,mask:List[int],is_training=True):
        state_tensor=torch.FloatTensor(observation).unsqueeze(0)
        mask_tensor=torch.BoolTensor(mask).unsqueeze(0)

        with torch.no_grad():
            logits,value=self.policy(state_tensor,mask_tensor)
            if is_training:
                dist=Categorical(logits=logits)
                action=dist.sample()
                log_prob=dist.log_prob(action)

                self.memory["states"].append(observation)
                self.memory["actions"].append(action.item())
                self.memory["log_probs"].append(log_prob.item())
                self.memory["values"].append(value.item())
                self.memory["masks"].append(mask)

                return action.item()
                
            else:
                action=torch.argmax(logits,dim=-1)
                return action.item()
            

    def learn(self):

        old_states = torch.FloatTensor(np.array(self.memory["states"]))
        old_actions = torch.LongTensor(self.memory["actions"])
        old_log_probs = torch.FloatTensor(self.memory["log_probs"])
        old_values = torch.FloatTensor(self.memory["values"])
        old_masks = torch.BoolTensor(np.array(self.memory["masks"]))
        
        rewards = self.memory["rewards"]
        dones = self.memory["dones"]

        advantages=[]
        gae=0

        for i in reversed(range(len(rewards))):

            if dones[i] or i==len(rewards)-1:
                next_value=0
                gae=0
            else:
                next_value=old_values[i+1]

            delta=rewards[i]+(gamma*next_value)-old_values[i]
            gae=delta+(gamma*lam*gae)

            advantages.insert(0,gae)

        advantages=torch.FloatTensor(advantages)
        returns=advantages + old_values

        advantages=(advantages-advantages.mean())/(advantages.std() +1e-7)

        for i in range(K_epoch):

            logits,curr_values=self.policy(old_states,old_masks)
            dists=Categorical(logits=logits)
            entropy=dists.entropy()
            
            new_log_probs=dists.log_prob(old_actions)

            ratios=torch.exp(new_log_probs-old_log_probs)

            surr1=ratios*advantages
            surr2=torch.clamp(ratios,1-eps_clip,1+eps_clip)*advantages

            loss_clip=-torch.min(surr1,surr2).mean()
            loss_vf=F.mse_loss(curr_values.squeeze(-1),returns)
            loss_e=-entropy.mean()

            final_loss=loss_clip + (c1*loss_vf) + (self.c2*loss_e)

            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

        self.c2=max(0.01,self.c2*0.995)
        self.clear_memory()

    
    def clear_memory(self):
        for key in self.memory:
            self.memory[key].clear()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)





