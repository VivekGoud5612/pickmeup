import os
from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Assuming your models are imported from the file we built earlier
from engine.agents.ppo.mappo_models import MAPPO_ACTOR, MAPPO_CRITIC


class Agent:
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        agent_id: int, 
        role: Any, 
        global_obs_dim: int = 54, 
        lr: float = 3e-4
    ):
        """
        The Master Orchestrator for a single multi-agent entity.
        Manages action inference, value estimation, and optimization steps.
        """
        self.agent_id = agent_id
        self.role = role

        # 1. Instantiate Core Brain Components
        # Each agent gets its own local policy (Actor) and centralized value tracker (Critic)
        self.actor = MAPPO_ACTOR(obs_dim, action_dim)
        self.critic = MAPPO_CRITIC(global_obs_dim)

        # Production Hook: Satisfies upstream framework patterns that look for self.policy
        self.policy = self.actor 

        # 2. Setup Separate Optimizers
        # We enforce eps=1e-5 to provide a safety buffer against floating-point zero errors
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

        # 3. Attach the Dynamic Statistics Tracker
        # Automatically keeps our training targets perfectly scaled between -1.0 and 1.0
        self.value_normalizer = ValueNormalizer()

        # 4. Ephemeral Trajectory Memories
        # Caches the immediate mathematical steps during environment execution 
        # so they can be neatly scraped by your global data logging system downstream.
        self.last_log_prob = 0.0
        self.last_value = 0.0

    def get_action(self, observation: list, action_mask: list, is_training: bool = True) -> int:
        """
        Processes local visions and returns a single, flattened discrete integer action choice.
        """
        # Convert raw lists into PyTorch execution vectors, adding a batch dimension [1, dim]
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)

        self.actor.eval() # Switch network context to freeze Batch Normalization/Dropout steps
        with torch.no_grad():
            # Generate the safe probability distribution using our zero-overhead action masker
            distribution = self.actor(obs_tensor, mask_tensor)
            
            if is_training:
                # Explore safely within the boundaries of allowed actions
                action = distribution.sample()
            else:
                # Play perfectly: pick the absolute highest probability option
                action = torch.argmax(distribution.probs, dim=-1)
                
            log_prob = distribution.log_prob(action)

        self.actor.train() # Restore network context back to training mode

        # Cache step signatures into local memory if this is an active training track
        if is_training:
            self.last_log_prob = log_prob.item()

        return int(action.item())

    def evaluate_global_state(self, global_state: list) -> float:
        """
        Queries the Centralized Critic to grade the overall security score of the board layout.
        """
        state_tensor = torch.FloatTensor(global_state).unsqueeze(0)
        
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(state_tensor)
        self.critic.train()
        
        # Cache and return the raw un-normalized value score
        self.last_value = value.item()
        return self.last_value

    def train_step(
        self, 
        local_obs: list, 
        global_states: list, 
        actions: list, 
        masks: list, 
        old_log_probs: list, 
        returns: list, 
        active_masks: list, 
        clip_epsilon: float = 0.2
    ) -> Tuple[float, float]:
        """
        Executes parallelized MAPPO optimizations using advanced Death Masking and Huber Loss.
        """
        # Convert incoming historic batches into full parallelized multi-dimensional GPU/CPU tensors
        local_obs_t = torch.FloatTensor(local_obs)
        global_states_t = torch.FloatTensor(global_states)
        actions_t = torch.LongTensor(actions)
        masks_t = torch.FloatTensor(masks)
        old_log_probs_t = torch.FloatTensor(old_log_probs)
        returns_t = torch.FloatTensor(returns)
        active_masks_t = torch.FloatTensor(active_masks) # Shape: [Batch], values: 1.0 = alive, 0.0 = dead

        # --- STEP 1: COMPUTE RUNTIME VALUE STATISTICS ---
        # Update our Welford normalizer matrices using the absolute truth returns from this batch
        self.value_normalizer.update(returns_t)
        normalized_returns = self.value_normalizer.normalize(returns_t)

        # --- STEP 2: OPTIMIZE CENTRALIZED CRITIC ---
        predicted_values = self.critic(global_states_t).squeeze(-1)
        
        # Turn the Critic's tiny scaled outputs back into real-world scores to calculate true advantage
        unnormalized_values = self.value_normalizer.denormalize(predicted_values)
        advantages = returns_t - unnormalized_values.detach()
        # Standardize the advantage vector to normalize variance across the team space
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Compute stable Huber Loss element-wise instead of crude Mean Squared Error
        critic_huber = F.huber_loss(predicted_values, normalized_returns, reduction='none')
        
        # DEATH MASKING: Multiply element-wise by active_masks. If an agent was dead, its loss becomes 0.
        # We divide by the count of living frames to prevent dead weight from diluting gradient strength.
        masked_critic_loss = (critic_huber * active_masks_t).sum() / torch.clamp(active_masks_t.sum(), min=1.0)

        # Execute dedicated Critic Backpropagation
        self.critic_optimizer.zero_grad()
        masked_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0) # Stop exploding gradients
        self.critic_optimizer.step()

        # --- STEP 3: OPTIMIZE DECENTRALIZED ACTOR ---
        distribution = self.actor(local_obs_t, masks_t)
        new_log_probs = distribution.log_prob(actions_t)

        # PPO Clipped Surrogate Objective Calculations
        ratios = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        raw_actor_loss = -torch.min(surr1, surr2)
        
        # DEATH MASKING: Ensure moves chosen by dead/inactive entities don't alter the policy weights
        masked_actor_loss = (raw_actor_loss * active_masks_t).sum() / torch.clamp(active_masks_t.sum(), min=1.0)

        # Execute dedicated Actor Backpropagation
        self.actor_optimizer.zero_grad()
        masked_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # Return clean loss telemetry integers for tracking in your logging dashboards
        return masked_actor_loss.item(), masked_critic_loss.item()

    def save(self, model_path: str):
        """Dumps Actor network parameters cleanly to disk checkpoints."""
        torch.save(self.actor.state_dict(), model_path)