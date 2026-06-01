from engine.agents.policy.rollout import RolloutBuffer
import torch
from torch.distributions import Categorical 
import torch.nn as nn
import torch.nn.functional as func 
from engine.agents.policy.network import Actor, Critic
from engine.agents.agent_data import AgentRole
import torch.optim as optim

LR = 1e-5
GAMMA = 0.95
LAMBDA = 0.95
K_EPOCH = 4
OBS_SIZE = 128
EPS_CLIP = 0.2
ACTION_SPACE_SIZE = 8
ENT_COEF = 0.1

class ValueNormalizer:
    def __init__(self, epsilon : float = 1e-5):

        self.epsilon = epsilon

        self.running_mean = 0.0
        self.running_var = 1.0

        self.count = epsilon   #Global count for step rewards. Initialized to epsilon,to prevent zero division on first update


    def update(self, returns : torch.Tensor) -> None:

        batch_mean = torch.mean(returns).item()
        batch_var = torch.var(returns).item()
        batch_count = returns.numel()

        total_count = self.count + batch_count

        # Difference between this batch's mean and old historical mean
        delta = batch_mean - self.running_mean

        # Shift the running mean towards the new batch mean
        self.running_mean += delta * (batch_count/total_count)

        # Welford's and Chad's Algorithm
        # For running variance we calculate the sum of differences for both old data(ma) and new batch(mb).
        # Then we combine them using a correction factor for how much the mean just shifted (delta**2)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count)/total_count

        # Convert M2 to running variance 
        self.running_var = M2 / total_count

        self.count = total_count


    def normalize(self, returns : torch.Tensor) -> torch.Tensor:
        # Standard deviation is square root of Variance
        std = (self.running_var + self.epsilon) ** 0.5

        return (returns - self.running_mean) / std
    

    def denormalize(self, values : torch.Tensor) -> torch.Tensor:

        std = (self.running_var + self.epsilon) ** 0.5

        return (values * std) + self.running_mean

class Trainer:

    def __init__(self, agent_id : int, role : AgentRole, device : torch.device = torch.device('cpu')):
        
        self.agent_id = agent_id 
        self.role = role 
        self.device = device 

        self.actor = Actor(OBS_SIZE, ACTION_SPACE_SIZE).to(self.device)
        self.critic = Critic(OBS_SIZE).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = LR, eps = 1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = LR, eps = 1e-5)

        self.value_normalizer = ValueNormalizer()

    @torch.no_grad()
    def get_action(self, obs_tensor : torch.Tensor, mask_tensor : torch.Tensor, is_training : bool = True) -> Tuple[torch.Tensor, torch.Tensor]:  ## Batched tensors here... of shape (num_envs, obs_dim) or (num_envs, mask_dim)

        self.actor.eval()   ## Switches network to evaluation mode, where it freezes BatchNorm or Dropout layers.
        distribution = self.actor(obs_tensor, mask_tensor)   ## Distribution of size (num_envs, num_actions)

        if is_training:
            actions = distribution.sample()  # returns tensors where each environment has one action so shape - (num_envs, 1)

        else:
            actions = torch.argmax(distribution.probs, dim = 1)   ## actions.shape = (num_envs, num_actions), and dim = 1 -> transform actions dim

        log_probs = distribution.log_prob(actions)
        self.actor.train()

        return actions, log_probs

    @torch.no_grad()
    def evaluate_global_state(self, state_tensor : torch.Tensor) -> torch.Tensor:   ## State_tensor shape is (num_envs, global_state_dim).. But if the global state dim is more than 1, then we need to flatten the dimension hard.

        self.critic.eval()
        values = self.critic(state_tensor).squeeze(-1)   ## normal values shape would be (num_envs, 1).. squeeze(-1) removes last dim.. so shape becomes (num_envs,)
        self.critic.train()

        return values

    
    def update_critic(self, batch : Dict[str, torch.Tensor]) -> float:
         
        #update critic for the whole environment at once - that is we use environment batch.
        global_states = batch['global_state']
        returns = batch['returns']

        
        self.value_normalizer.update(returns)
        normalized_returns = self.value_normalizer.normalize(returns)

        predicted_values = self.critic(global_states).squeeze(-1)
        critic_loss = func.huber_loss(predicted_values, normalized_returns)

        self.critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm(self.critic.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smoothen out the gradient. Although we use adam and such for updative learning rate this clipping helps so..
        self.critic.optimizer.step()

        return critic_loss 

    
    def update_actor(self, batch : Dict[str, torch.Tensor]) -> float:

        distribution = self.actor(batch['obs'], batch['action_masks'])
        new_log_probs = distribution.log_prob(batch['actions'])
        entropy = distribution.entropy()

        ratios = torch.exp(new_log_probs - batch['log_probs'])   ## New Log probs are also the size of batch.. because actions are sampled from the same batch
        surr1 = ratios * batch['advantages']
        surr2 = torch.clamp(ratios, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * batch['advantages']
        raw_policy_loss = -torch.min(surr1, surr2)  ## We are aiming to maximize surr1 or 2 which increase the ratios and advantages which is good.


        agent_loss = raw_policy_loss - (ENT_COEF * entropy)  ## Also aim to maximise entropy (weighted) to let agents explore more.

        active_masks = batch['active_masks']
        masked_actor_loss = (agent_loss * active_masks).sum() / torch.clamp(active_masks, min = 1.0)

        self.actor_optimizer.zero_grad()
        masked_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        return masked_actor_loss.item()