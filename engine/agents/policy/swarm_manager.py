import torch
from engine.agents.policy.network import Actor, Critic
from engine.utils.enums import AgentRole
import torch.optim as optim
from typing import List, Dict, Any
import torch.nn as nn

class SwarmManager:  ## Normal class and is responsible for running 4 different agents.. and make this according to the trainer so that we cna use tha tdirectly.....

    def __init__(self, base_obs_dim : int = 24, hero_intent_dim : int = 24, global_obs_dim : int = 96, actions_dim : dict = None, alr : float = 3e-4, clr : float = 1e-3):  ### There are three heroes and their itnents get broadcasted to all agents after one pass...
       # base_obs_dim = base_obs_dim + hero_intent_dim

        self.tank_actor = Actor(base_obs_dim)
        self.dealer_actor = Actor(base_obs_dim)
        self.healer_actor = Actor(base_obs_dim)

        self.boss_actor = Actor(base_obs_dim)

        self.tank_critic = Critic(global_obs_dim) ### For now all the action spaces are the same..
        self.dealer_critic = Critic(global_obs_dim)
        self.healer_critic = Critic(global_obs_dim)

        self.boss_critic = Critic(global_obs_dim)

        ##OPTIMIZERS.. We group the parameters inside one singe optimizer as that can easily differentiate between the losses and update weights accordingly
        self.tank_aoptim = optim.Adam(self.tank_actor.parameters(), lr = alr)
        self.dealer_aoptim = optim.Adam(self.dealer_actor.parameters(), lr = alr)
        self.healer_aoptim = optim.Adam(self.healer_actor.parameters(), lr = alr)
        self.boss_aoptim = optim.Adam(self.boss_actor.parameters(), lr = alr)

        self.tank_coptim = optim.Adam(self.tank_critic.parameters(), lr = clr)
        self.dealer_coptim = optim.Adam(self.dealer_critic.parameters(), lr = clr)
        self.healer_coptim = optim.Adam(self.healer_critic.parameters(), lr = clr)
        self.boss_coptim = optim.Adam(self.boss_critic.parameters(), lr = clr)

    def to(self, device):
        self.device = device
        self.tank_actor.to(device)
        self.healer_actor.to(device)
        self.dealer_actor.to(device)
        self.boss_actor.to(device)
        self.tank_critic.to(device)
        self.healer_critic.to(device)
        self.dealer_critic.to(device)
        self.boss_critic.to(device)
        self.device = device
        return self  ## Return self so that we cna store the same object inside that object again after pushing all these to GPU memory

    def get_actions_and_values(self, obs : torch.Tensor, global_state : torch.Tensor, action_masks : torch.Tensor, is_training : bool = True):  ## Arrays are sent to GPU memory in trainer so we cna rest easy

        self.tank_actor.eval()
        self.healer_actor.eval()
        self.dealer_actor.eval()
        self.boss_actor.eval()
        self.tank_critic.eval()
        self.healer_critic.eval()
        self.dealer_critic.eval()
        self.boss_critic.eval()

        num_envs = obs.shape[0] ## To just get the number of envs ...

        tank_obs = obs[:, AgentRole.TANK, :]  ## Shape (num_envs, 24)
        dealer_obs = obs[:, AgentRole.DEALER, :]
        healer_obs = obs[:, AgentRole.HEALER, :]
        boss_obs = obs[:, AgentRole.BOSS, :]

        tank_global = global_state[:, AgentRole.TANK, :] 
        dealer_global = global_state[:, AgentRole.DEALER, :]
        healer_global = global_state[:, AgentRole.HEALER, :]
        boss_global = global_state[:, AgentRole.BOSS, :]

        tank_action_masks = action_masks[:, AgentRole.TANK, :] ## Shape (num_envs, 8)
        dealer_action_masks = action_masks[:, AgentRole.DEALER, :]
        healer_action_masks = action_masks[:, AgentRole.HEALER, :]
        boss_action_masks = action_masks[:, AgentRole.BOSS, :]

        boss_dist = self.boss_actor(boss_obs, boss_action_masks) ## No need for two pass here as boss is a single agent..

        ## DUMMY INTENT PASS AND FIRST PASS
        dummy_hero_intent = torch.zeros((num_envs, 24), device = self.device) ## Create a dummy hero intent, so that we can start our first pass

        #tank_dist = self.tank_actor(torch.cat([tank_obs, dummy_hero_intent], dim = -1), tank_action_masks)  ## Tank intent.. first pass..
        #dealer_dist = self.dealer_actor(torch.cat([dealer_obs, dummy_hero_intent], dim = -1), dealer_action_masks)  ## concat along the last dimensions.. that is keep addign elements along the last dim
        #healer_dist = self.healer_actor(torch.cat([healer_obs, dummy_hero_intent], dim = -1), healer_action_masks)  # As that is a distribution we take the values of those distributions namely logits with dist.logits attribute

        ### TEAM BROADCAST - Heroes share the intent to all the other heroes in the same team.. And instead of logits which are sometimes -inf, and that into 0 gives us nan.. so we use probs
        #hero_intent_broadcast = torch.cat([tank_dist.probs, dealer_dist.probs, healer_dist.probs], dim = -1).detach() ## Detach removes this tensor from pytorchs computation graph... (detaches the tensor from automatic differentiation..)

        tank_dist_final = self.tank_actor(tank_obs, tank_action_masks)
        dealer_dist_final = self.dealer_actor(dealer_obs, dealer_action_masks)
        healer_dist_final = self.healer_actor(healer_obs, healer_action_masks)

        tank_value = self.tank_critic(tank_global) 
        dealer_value = self.dealer_critic(dealer_global)
        healer_value = self.healer_critic(healer_global)
        boss_value = self.boss_critic(boss_global)

        return (
            (tank_dist_final, dealer_dist_final, healer_dist_final, boss_dist),
            (tank_value, dealer_value, healer_value, boss_value),
            (dummy_hero_intent)  ## So that we can store that in the rollout buffer
        )
        

    def get_value(self, global_state : torch.Tensor, role : AgentRole):

        if role == AgentRole.TANK:
            values = self.tank_critic(global_state)
        elif role == AgentRole.DEALER:
            values = self.dealer_critic(global_state)
        elif role == AgentRole.HEALER:
            values = self.healer_critic(global_state)
        else:
            values = self.boss_critic(global_state)

        return values  ## That is it.. simple but the actual problem is actor.. Now do we return the distributions based on intent or what ??

    def get_dist(self, total_obs : torch.Tensor, action_masks : torch.Tensor, role : AgentRole):  ## Let us store the intents at that time for the batch as well.. in the buffer

        #total_obs = torch.cat([obs, intents], dim = -1)  ## For the whole batch we add stored intents to the stored obs

        if role == AgentRole.TANK:
            dist = self.tank_actor(total_obs, action_masks)
        elif role == AgentRole.DEALER:
            dist = self.dealer_actor(total_obs, action_masks)
        elif role == AgentRole.HEALER:
            dist = self.healer_actor(total_obs, action_masks)
        else:
            dist = self.boss_actor(total_obs, action_masks)

        return dist


    def update_weights(self, actor_loss : Dict[str, torch.Tensor], critic_loss : Dict[str, torch.Tensor]): ## Get those two losses per epoch and keep updating the weights..

        self.tank_actor.train()
        self.healer_actor.train()
        self.dealer_actor.train()
        self.boss_actor.train()
        self.tank_critic.train()
        self.healer_critic.train()
        self.dealer_critic.train()
        self.boss_critic.train()

        self.tank_aoptim.zero_grad() ## Reset gradients..
        actor_loss[AgentRole.TANK].backward() ## calculate gradients
        nn.utils.clip_grad_norm_(self.tank_actor.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smooth the gradients     
        self.tank_aoptim.step()  ## Update weights

        self.dealer_aoptim.zero_grad() ## Reset gradients..
        actor_loss[AgentRole.DEALER].backward() ## calculate gradients
        nn.utils.clip_grad_norm_(self.dealer_actor.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smooth the gradients     
        self.dealer_aoptim.step()  ## Update weights

        self.healer_aoptim.zero_grad() ## Reset gradients..
        actor_loss[AgentRole.HEALER].backward() ## calculate gradients
        nn.utils.clip_grad_norm_(self.healer_actor.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smooth the gradients     
        self.healer_aoptim.step()  ## Update weights

        self.boss_aoptim.zero_grad() ## Reset gradients..
        actor_loss[AgentRole.BOSS].backward() ## calculate gradients
        nn.utils.clip_grad_norm_(self.boss_actor.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smooth the gradients     
        self.boss_aoptim.step()  ## Update weights

        ##CRITICS
        self.tank_coptim.zero_grad() ## Reset gradients..
        critic_loss[AgentRole.TANK].backward() ## calculate gradients
        nn.utils.clip_grad_norm_(self.tank_critic.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smooth the gradients     
        self.tank_coptim.step()  ## Update weights

        self.dealer_coptim.zero_grad() ## Reset gradients..
        critic_loss[AgentRole.DEALER].backward() ## calculate gradients
        nn.utils.clip_grad_norm_(self.dealer_critic.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smooth the gradients     
        self.dealer_coptim.step()  ## Update weights

        self.healer_coptim.zero_grad() ## Reset gradients..
        critic_loss[AgentRole.HEALER].backward() ## calculate gradients
        nn.utils.clip_grad_norm_(self.healer_critic.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smooth the gradients     
        self.healer_coptim.step()  ## Update weights

        self.boss_coptim.zero_grad() ## Reset gradients..
        critic_loss[AgentRole.BOSS].backward() ## calculate gradients
        nn.utils.clip_grad_norm_(self.boss_critic.parameters(), max_norm = 10.0)   ## we clip the update to max update of 10 to smooth the gradients     
        self.boss_coptim.step()  ## Update weights

