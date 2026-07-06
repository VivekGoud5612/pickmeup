import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func
from typing import Dict, Tuple
import numpy as np

# Assuming you import your networks and ValueNormalizer here
from engine.agents.policy.network import SharedActor, SharedCritic
# from engine.agents.policy.utils import ValueNormalizer 

class MAPPOAgent:
    def __init__(self, device: torch.device, lr_actor: float = 3e-4, lr_critic: float = 1e-3):
        self.device = device
        
        # 1. Initialize Decoupled Networks
        self.actor = SharedActor().to(device)
        self.critic = SharedCritic().to(device)
        
        # 2. Independent Optimizers (The key to preventing the Critic Bully effect)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
        
        self.value_normalizer = ValueNormalizer()
        
        # Hyperparameters
        self.eps_clip = 0.2
        self.ent_coef = 0.01
        self.max_grad_norm = 10.0

    @torch.no_grad()
    def get_actions_and_values(self, obs: np.ndarray, global_state: np.ndarray, 
                               roles: np.ndarray, action_masks: np.ndarray, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lightning-fast batched inference for Rollout Collection"""
        self.actor.eval()
        self.critic.eval()
        
        E, A = roles.shape
        
        # Flatten batch dimensions for maximum GPU throughput
        t_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(E * A, -1)
        t_global = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).view(E * A, -1)
        t_roles = torch.as_tensor(roles, dtype=torch.long, device=self.device).view(E * A)
        t_masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device).view(E * A, -1)

        # --- Actor Pass ---
        dist = self.actor(t_obs, t_roles, t_masks)
        
        if is_training:
            actions = dist.sample()
        else:
            actions = torch.argmax(dist.probs, dim=-1)
            
        log_probs = dist.log_prob(actions)

        # --- Critic Pass ---
        norm_values = self.critic(t_global, t_roles)
        # Denormalize immediately so the RolloutBuffer stores raw values for accurate PBRS math
        raw_values = self.value_normalizer.denormalize(norm_values)

        # Reshape back to Environment Format (E, A) and kick back to CPU NumPy
        return (actions.view(E, A).cpu().numpy(), 
                log_probs.view(E, A).cpu().numpy(), 
                raw_values.view(E, A).cpu().numpy())

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Runs the PPO Update on a randomized, flattened mini-batch"""
        self.actor.train()
        self.critic.train()

        # Unpack batch (Already flattened to 1D/2D by RolloutBuffer generator)
        b_obs = batch['obs'].to(self.device)
        b_global = batch['global_state'].to(self.device)
        b_roles = batch['roles'].to(self.device)
        b_actions = batch['actions'].to(self.device)
        b_log_probs = batch['log_probs'].to(self.device)
        b_advs = batch['advantages'].to(self.device)
        b_returns = batch['returns'].to(self.device)
        b_act_masks = batch['action_masks'].to(self.device)
        b_active_masks = batch['active_masks'].to(self.device) # Shape: (batch_size,)

        # Safe divisor for active masking to prevent div-by-zero if everyone is dead in this batch
        active_sum = torch.clamp(b_active_masks.sum(), min=1.0)

        # ==========================================
        # 1. CRITIC UPDATE (Isolated)
        # ==========================================
        self.value_normalizer.update(b_returns)
        norm_returns = self.value_normalizer.normalize(b_returns)
        
        pred_values = self.critic(b_global, b_roles)
        raw_critic_loss = func.huber_loss(pred_values, norm_returns, reduction='none')
        
        # Apply Active Mask (Ignore dead agents)
        critic_loss = (raw_critic_loss * b_active_masks).sum() / active_sum

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # ==========================================
        # 2. ACTOR UPDATE (Isolated)
        # ==========================================
        # Normalize advantages at the mini-batch level for stability
        b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        dist = self.actor(b_obs, b_roles, b_act_masks)
        new_log_probs = dist.log_prob(b_actions)
        entropy = dist.entropy()

        ratios = torch.exp(new_log_probs - b_log_probs)
        surr1 = ratios * b_advs
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * b_advs
        
        raw_actor_loss = -torch.min(surr1, surr2) - (self.ent_coef * entropy)
        
        # Apply Active Mask (Ignore dead agents so they don't corrupt the policy)
        actor_loss = (raw_actor_loss * b_active_masks).sum() / active_sum

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(), 
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item()
        }