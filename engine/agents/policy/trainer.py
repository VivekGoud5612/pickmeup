from engine.agents.policy.rollout import RolloutBuffer
import torch
from torch.distributions import Categorical 
import torch.nn as nn
import torch.nn.functional as func 
from engine.agents.policy.network import Actor, Critic
from engine.agents.agent_data import AgentRole
import torch.optim as optim
from engine.environmen.state import GameState
from engine.environment.observation import ObservationBuilder

LR = 3e-4
GAMMA = 0.95
LAMBDA = 0.95
K_EPOCH = 4
EPS_CLIP = 0.2
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



class Agent:

    def __init__(self, agent_id : int, role : AgentRole, device : torch.device = "cpu"): ## As we only use this for get action and evaluate global state, there is no need for GPU.. and the tensors inside that batch of update actor are on the GPU. we need to write that in main file
        
        self.agent_id = agent_id 
        self.role = role 
        self.device = device     

        self.actor = Actor(ObservationBuilder.OBS_SIZE, GameState.NUM_ACTIONS).to(self.device)  #Role embedding size is already written or defined there..
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = LR, eps = 1e-5)

    @torch.no_grad()
    def get_action(self, obs_tensor : np.ndarray, role_id_tensor : np.ndarray, mask_tensor : np.ndarray, is_training : bool = True) -> Tuple[torch.Tensor, torch.Tensor]:  ## Batched tensors here... of shape (num_envs, obs_dim) or (num_envs, mask_dim)

        obs_tensor = torch.as_tensor(obs_tensor, device = self.device)
        role_id_tensor = torch.as_tensor(role_id_tensor, device = self.device)
        mask_tensor = torch.as_tensor(mask_tensor, device = self.device)

        self.actor.eval()   ## Switches network to evaluation mode, where it freezes BatchNorm or Dropout layers.
        distribution = self.actor(obs_tensor, role_id_tensor, mask_tensor)   ## Distribution of size (num_actions,) .. As we separate out actions, observations and masks...

        if is_training:
            actions = distribution.sample()  # returns tensors where each environment has one action so shape - (1)  for each environment

        else:
            actions = torch.argmax(distribution.probs, dim = 1)   ## actions.shape = (num_envs, ), and dim = 1 -> transform actions dim and takes the maximum from that 8 space action dim for each agent. As we take observation, and mask for each agent the actual output of action.shape is (num_envs, )

        log_probs = distribution.log_prob(actions)
        self.actor.train()

        return actions, log_probs
    
    def update_actor(self, batch : Dict[str, torch.Tensor]) -> float:  # In the training loop we need to pass agent ID to the buffer as well to get that specific agents batch.. of shape (batch_size, shape of whatever that other metric)

        distribution = self.actor(batch['obs'], batch['local_ids'], batch['action_masks'])  ## For our updated critic, distribution of size (batch_size, num_actions = 8)
        new_log_probs = distribution.log_prob(batch['actions'])  # Shape (batch_size, num_actions = 8)
        entropy = distribution.entropy() # (batch_size, 1)

        ratios = torch.exp(new_log_probs - batch['log_probs'])   ## New Log probs are also the size of batch.. because actions are sampled from the same batch
        surr1 = ratios * batch['advantages'] # advantages of size (batch, num_act)
        surr2 = torch.clamp(ratios, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * batch['advantages']
        raw_policy_loss = -torch.min(surr1, surr2)  ## We are aiming to maximize surr1 or 2 which increase the ratios and advantages which is good.

        agent_loss = raw_policy_loss - (ENT_COEF * entropy)  ## Also aim to maximise entropy (weighted) to let agents explore more.

        active_masks = batch['active_masks']
        masked_actor_loss = (agent_loss * active_masks).sum() / torch.clamp(active_masks.sum(), min = 1.0)  ## refer notes

        self.actor_optimizer.zero_grad()
        masked_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        return masked_actor_loss.item()


class Global:

    def __init__(self, device : str = "cpu"):

        self.critic = Critic(ObservationBuilder.GLOBAL_OBS_SIZE).to(self.device)
        self.value_normalizer = ValueNormalizer()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = LR, eps = 1e-5)

    @torch.no_grad()
    def evaluate_global_state(self, state_tensor : np.ndarray) -> torch.Tensor:   ## State_tensor shape is (num_envs, global_state_dim).. But if the global state dim is more than 1, then we need to flatten the dimension hard.

        state_tensor = torch.as_tensor(state_tensor, device = self.device)

        self.critic.eval()
        normalized_values = self.critic(state_tensor).squeeze(-1)   ## normal values shape would be (num_envs, 1).. squeeze(-1) removes last dim.. so shape becomes (num_envs,)
        self.critic.train()
        ## Note that as the target is normalized and all, critics weights change such a way that , values come out normalized from the network.. so we have to denormalize those as to compute returns and advantages, else the value signal becomes weak which might cause advantages or returns to skew towards current rewards

        raw_values = self.value_normalizer.denormalize(normalized_values)
        return raw_values 
    
    
    def update_critic(self, batch : Dict[str, torch.Tensor]) -> float:
         
        #update critic for the whole environment at once - that is we use environment batch.
        global_states = batch['global_state']  # Shape (batch_size, global_state[size of 96]..)
        global_ids = batch['global_ids']  ## Shape (batch_size, 4)
        returns = batch['returns']   # Shape (batch_size, )... because we mashed all the agents, envs and steps into one giant flat size and we sampled the batch size from that..

        self.value_normalizer.update(returns)
        normalized_returns = self.value_normalizer.normalize(returns)   ## size (batch_size,)

        predicted_values = self.critic(global_states, global_ids).squeeze(-1)  # output was (batch_size, 1) -> (batch_size,)
        critic_loss = func.huber_loss(predicted_values, normalized_returns)

        self.critic_optimizer.zero_grad()   ### Our critic object is self.critic, but here we are underscoring optimizer which is another object entirely.. so lets see
        critic_loss.backward()
        nn.utils.clip_grad_norm(self.critic.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smoothen out the gradient. Although we use adam and such for updative learning rate this clipping helps so..
        self.critic_optimizer.step()

        return critic_loss 