import torch
import torch.nn.functional as func ## Functional neural network utilities like .mse_loss()
import torch.optim as optim 
from torch.distributions import Categorical  ##Creates probabilites distributions over discrete actions
import numpy as np
from typing import List 
from engine.agents.ppo.model import ActorCritic
from engine.environment.observation import Observation

LR = 3e-4   ##Learning Rate
GAMMA = 0.99   ## Discounted Factor for future value.
LAMBDA = 0.95  ## Discounted factor for future advantage function (BIAS and VARIANCE Tradeoff)
C1 = 0.5   ##Weight for critic loss... Need to understand more
EPS_CLIP = 0.2   ##PPO clipping range
K_EPOCH = 4  ## reuse rollout data 4 times

class AgentPolicy:
    def __init__(self, obs_size : int, action_size : int):
        self.action_size = action_size 

        self.c2 = 0.1 ## Entropy coefficient.. controls exploration strength
        self.policy = ActorCritic(obs_size, action_size)
        self.optimizer = optim.Adam(parameters = self.policy.parameters(), lr = LR)

        self.memory = {
            'states'  : [],
            'actions' : [],
            'log_probs' : [],
            'values' : [],
            'rewards' : [],
            'masks' : [],
            'dones' : [],
        }

    def store_reward(self, reward : float, done : bool):
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
    
    def get_action(self, observation : Observation, mask : List[int], is_training = True):
        obs_vector = observation.to_vector()

        state_tensor = torch.FloatTensor(obs_vector).unsqueeze(0)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.policy(state_tensor, mask_tensor)

            if is_training:
                dist = Categorical(logits = logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                self.memory['states'].append(obs_vector)
                self.memory['actions'].append(action.item())  ## .item() is the get the value stored inside
                self.memory['log_probs'].append(log_prob.item())
                self.memory['values'].append(value.item())
                self.memory['masks'].append(mask)

                return action.item()
            
            else:
                action = torch.argmax(logits, dim = -1)
                return action.item()

    def learn(self):

        old_states = torch.FloatTensor(np.array(self.memory['states']))
        old_actions = torch.LongTensor(self.memory['actions'])
        old_log_probs = torch.FloatTensor(self.memory['log_probs'])
        old_values = torch.FloatTensor(self.memory['values'])
        old_masks = torch.BoolTensor(np.array(self.memory['masks']))

        rewards = self.memory['rewards']
        dones = self.memory['dones']

        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):

            if dones[i] or i == len(rewards) - 1:
                next_value = 0
                gae = 0
            else:
                next_value = old_values[i+1]

            delta = rewards[i] + (GAMMA*next_value) - old_values[i]
            gae = delta + (GAMMA*LAMBDA*gae)

            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + old_values

        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-7)

        for i in range(K_EPOCH):  ## Learn 4 times for each data ...

            logits, current_values = self.policy(old_states, old_masks)  ##Logits and Values for all the previos memory at once... So we get 2048, 7, 1 size of actions
            dists = Categorical(logits = logits)
            entropy = dists.entropy()  # We get entropy from the same distribution

            new_log_probs = dists.log_prob(old_actions)

            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            loss_clip = -torch.min(surr1, surr2).mean()
            loss_vf = func.mse_loss(curr_values.squeeze(-1), returns)
            loss_e = -entropy.mean()

            final_loss = loss_clip + (C1 * loss_vf) + (self.c2 * loss_e)

            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

        self.c2 = max(0.01, self.c2 * 0.995) ## Decay c2 per step, to reduce entropy at the importance of entropy at the very last
        self.clear_memory()

    def clear_memory(self):
        for key in self.memory.keys():
            self.memory[key].clear()
