from engine.agents.policy.rollout import RolloutBuffer
import torch
from torch.distributions import Categorical 
import torch.nn as nn
import torch.nn.functional as func 
from engine.agents.policy.network import Actor, Critic
import torch.optim as optim
from engine.environment.state import GameState
from engine.environment.observation import ObservationBuilder
import numpy as np
from typing import Tuple, Dict
from engine.agents.policy.normalizer import ValueNormalizer

GAMMA = 0.95
LAMBDA = 0.95
K_EPOCH = 4
EPS_CLIP = 0.2
ENT_COEF = 0.1


class MAgent:

    def __init__(self, device : torch.device, lr_actor : float = 3e-4, lr_critic : float = 1e-3):

        self.device = device 
        self.actor = Actor(ObservationBuilder.OBS_SIZE, GameState.NUM_ACTIONS).to(self.device)  #Role embedding size is already written or defined there..
        self.critic = Critic(ObservationBuilder.GLOBAL_STATE_SIZE).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = lr_actor, eps = 1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr_critic, eps = 1e-5)

        self.eps_clip = 0.2  ## Epsilon clipping , used at last... for surr 2 that is to limit the actor updates
        self.max_grad_norm = 10.0

        self.value_normalizer = ValueNormalizer()

    @torch.no_grad()
    def get_actions_and_values(self, obs: np.ndarray, global_state: np.ndarray, roles: np.ndarray, action_masks: np.ndarray, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Natively multi-dimensional inference for Rollout Collection"""
        self.actor.eval()
        self.critic.eval()
        
        # NO MORE FLATTENING. Pass the exact environment tracking shapes:
        # obs: (num_envs, num_agents, 24) | global_state: (num_envs, num_agents, 96)
        # roles: (num_envs, num_agents)  | action_masks: (num_envs, num_agents, 8)
        t_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        t_global = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        t_roles = torch.as_tensor(roles, dtype=torch.long, device=self.device)
        t_masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)

        # --- Actor Pass (Preserves leading dimensions automatically) ---
        dist = self.actor(t_obs, t_roles, t_masks)
        
        if is_training:
            actions = dist.sample()            # Shape: (num_envs, num_agents)
        else:
            actions = torch.argmax(dist.probs, dim=-1) # Shape: (num_envs, num_agents)
            
        log_probs = dist.log_prob(actions)    # Shape: (num_envs, num_agents)

        # --- Critic Pass ---
        norm_values = self.critic(t_global, t_roles) # Shape: (num_envs, num_agents)
        raw_values = self.value_normalizer.denormalize(norm_values)  ## ## Note that as the target is normalized and all, critics weights change such a way that , values come out normalized from the network.. so we have to denormalize those as to compute returns and advantages, else the value signal becomes weak which might cause advantages or returns to skew towards current rewards

        # Return clean arrays straight back to CPU NumPy with zero overhead
        return actions.cpu().numpy(), log_probs.cpu().numpy(), raw_values.cpu().numpy()


    def update(self, batch : Dict[str, torch.Tensor], ent_coef : float) -> Dict[str, float]:  # A single update method for a single batch by a single batch creator...

        self.actor.train()  ## In training mode...
        self.critic.train() ##..Same

        ### Unpacking batch elements
        b_obs = batch['obs'].to(self.device)                 # Shape: (mini_batch_size, 24)
        b_global = batch['global_state'].to(self.device)     # Shape: (mini_batch_size, 96)
        b_roles = batch['role_ids'].to(self.device)             # Shape: (mini_batch_size,)
        b_actions = batch['actions'].to(self.device)         # Shape: (mini_batch_size,)
        b_log_probs = batch['log_probs'].to(self.device)     # Shape: (mini_batch_size,)
        b_advs = batch['advantages'].to(self.device)         # Shape: (mini_batch_size,)
        b_returns = batch['returns'].to(self.device)         # Shape: (mini_batch_size,)  ## There is no need for values here.. we try to bring in critic predicted values as close to returns (advantages + values)
        b_action_masks = batch['action_masks'].to(self.device)   # Shape: (mini_batch_size, 8)
        b_active_masks = batch['active_masks'].to(self.device) # Shape: (mini_batch_size,)

        active_sum = torch.clamp(b_active_masks.sum(), min = 1.0)  ## Sum of all aliveagents..

        ## CRITIC UPDATE 
        self.value_normalizer.update(b_returns)  ## Calculate the running mean and all
        normalized_returns = self.value_normalizer.normalize(b_returns) ## Shape (batch,)

        pred_values = self.critic(b_global, b_roles)  ## Note that b_roles is the same for both actor and critic
        raw_critic_loss = func.huber_loss(pred_values, normalized_returns)

        self.critic_optimizer.zero_grad()   ### Our critic object is self.critic, but here we are underscoring optimizer which is another object entirely.. so lets see
        raw_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm = self.max_grad_norm)   ## we clip the update to max update of 10 to smoothen out the gradient. Although we use adam and such for updative learning rate this clipping helps so..
        self.critic_optimizer.step()


        ### ACTOR UPDATE 
        b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8) ## A small time normalization of the advantages, so that we get small surr 1 and surr 2.. so that those things stabilize..

        dist = self.actor(b_obs, b_roles, b_action_masks)  ## We calculate the action distributions again for the same obs, roles which are already in the buffer
        new_log_probs = dist.log_prob(b_actions) ## And new logs to the same... I guess for the updated network these would change... So we wanted to compare old probs and new probs
        entropy = dist.entropy()
        
        ratios = torch.exp(new_log_probs - b_log_probs)  ## Comparing policy.. that is log probs
        surr1 = ratios * b_advs
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advs ## Only limit the ratio to be somewhat within a range..

        raw_actor_loss = -torch.min(surr1, surr2) - (ent_coef * entropy) ## Maximise both entropy and surr , that is ratio into advantage, that si the new policy should be better htan old and also advantages should be large.. that is how much good is the current action than the past...
        actor_loss = ((raw_actor_loss * b_active_masks) / active_sum).sum()  ## Just sum all those losses for alive agents

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(), 
            "critic_loss": raw_critic_loss.item(),
            "entropy": entropy.mean().item()
        }
