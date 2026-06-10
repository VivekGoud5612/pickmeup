import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List,Any,Tuple
from .model import MAPPO_ACTOR,MAPPO_CRITIC
from engine.agents.ppo.normalizer import Value_Normalizer




class Agent:
    def __init__(
                self, 
                local_obs_dim : int, 
                actions_dim : int, 
                agent_id : int, 
                role : Any, 
                global_obs_dim : int, 
                hidden_dim : int, 
                lr : float = 3e-4
                ):

        self.local_obs_dim = local_obs_dim
        self.actions_dim = actions_dim

        self.agent_id  = agent_id
        self.role = role
        self.c2 = 0.1

        self.actor = MAPPO_ACTOR(local_obs_dim, actions_dim, hidden_dim)
        self.critic = MAPPO_CRITIC(global_obs_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = lr, eps = 1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr, eps = 1e-5)

        self.value_normalizer = Value_Normalizer()

        #Stores immediate log_probes and values to be scraped by global buffer later
        self.last_log_prob = 0.0
        self.last_value = 0.0


    def get_action(self, observation : List, action_mask : List, is_training : bool = True) -> int:

        # Convert list to float tensors and add extra dimension at the front since nn.Linear expects batches
        #Removed unsqueeze since we are implementing vectorized envs,so obs_tensor already is a batch
        obs_tensor = torch.FloatTensor(observation)
        mask_tensor = torch.FloatTensor(action_mask)

        self.actor.eval()   #Switch network to freeze Batch normalization/Dropout steps.Uses 100 percent network and gives deterministic answers
        with torch.no_grad():
            
            #The actor network also returns a 2D matrix of [num_envs, action_dim] ,list of lists
            #But Categorical gives probs,and sample selects one action per num_env 
            #So in sample() it returns a tensor of 1D array,we can return by copying to cpu and converting to a np array
            distribution = self.actor(obs_tensor, mask_tensor)

            if is_training:
                action = distribution.sample()

            else:
                action = torch.argmax(distribution.probs, dim = -1)

            log_prob = distribution.log_prob(action)
        
        self.actor.train()  #Switch back to training mode

        #Stores the immediate log_prob for the action taken.
        if is_training:
            self.last_log_prob = log_prob

        return action.cpu().numpy()    #return the full batch array
    

    def evaluate_global_state(self, global_state : List) -> float:
        #Grades the overall global state ... i.e State Value

        state_tensor = torch.FloatTensor(global_state)

        #The network returns a 2D matrix [num_envs,value_dim] ,list of lists
        #But we need 1D array for our buffer storage,so squeeze the last useless dimension
        #In actor due to Categorical,there was no need
        #Copy that tensor to cpu and convert to numpy
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(state_tensor).squeeze(-1)   

            denormalized_value = self.value_normalizer.Denormalize(value)
        
        self.critic.train()

        #Stores the denormalized value for the global state 
        self.last_value = denormalized_value.cpu().numpy()

        return self.last_value
    

    def learn(
            self,
            local_obs : List,
            global_states : List,
            actions : List,
            masks : List,
            old_log_probs : List,
            returns : List,
            advantages : List,
            active_masks : List,
            eps_clip : float = 0.2
        ) -> Tuple[float, float]:

        #Convert incoming batch lists into tensors
        local_obs_t = torch.FloatTensor(local_obs)
        global_states_t = torch.FloatTensor(global_states)
        actions_t = torch.LongTensor(actions)   #Since actions are expected to be integers in Categorical log prob calculations
        masks_t = torch.FloatTensor(masks)
        old_log_probs_t = torch.FloatTensor(old_log_probs)
        returns_t = torch.FloatTensor(returns)
        advantages_t = torch.FloatTensor(advantages)
        active_masks_t = torch.FloatTensor(active_masks)  #Value : 1.0 if alive , else 0.0

        #Update our Welford value normalizer using this batch's data
        self.value_normalizer.update(returns_t)

        #Optimize Centralized Critic
        normalized_returns = self.value_normalizer.Normalize(returns_t)

        #Calculate the predicted values for the global_states from critic and squueze the last dimension,which results in a 1D array of predicted state values
        predicted_values = self.critic(global_states_t).squeeze(-1)


        #Calculate Critic's Huber Loss according to normalized returns and predicted values
        #Reduction='None' since by default if reduction is not specified,reduction='Mean',Pytorch collapses the loss into a single number=mean
        #But we need the total loss matrix to do our death masking ,so reduction = 'None'
        critic_huber_loss = F.huber_loss(predicted_values, normalized_returns, reduction = 'none')

        #Apply death masking 
        #only add the loss of alive agents and divide by the number of alive agents,to prevent dead weight from diluting gradient strength
        #Dead agents backpropagation does nto include the dummy data when it si dead,which protects its learned network
        masked_critic_loss = (critic_huber_loss * active_masks_t).sum() / torch.clamp(active_masks_t.sum(), min = 1.0)

        #Execute Critic Backpropagation
        self.critic_optimizer.zero_grad()
        masked_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm = 10.0)   #Stop exploding gradients before step
        self.critic_optimizer.step()

        #Normalize the advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-5)

        #Optimize Decentralized Actor
        new_distributions = self.actor(local_obs_t, masks_t)
        new_log_probs = new_distributions.log_prob(actions_t)

        #Calculate PPO Surrogate Functions
        ratios = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratios * advantages_t
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip)*advantages_t

        #Removed mean(), again we need the total loss matrix for death masking
        actor_loss = -torch.min(surr1, surr2)

        #Apply death masking to actor loss
        masked_actor_loss = (actor_loss * active_masks_t).sum() / torch.clamp(active_masks_t.sum(), min = 1.0)

        #Execute Actor Backpropagation
        self.actor_optimizer.zero_grad()
        masked_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm = 10.0)    #Stop exploding gradients before step
        self.actor_optimizer.step()


        return masked_actor_loss.item(), masked_critic_loss.item()
    


    def save(self, model_path):
        #Saves actor network parameters into model path
        torch.save(self.actor.state_dict(), model_path)