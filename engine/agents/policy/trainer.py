from engine.agents.policy.rollout import RolloutBuffer
import torch
from torch.distributions import Categorical 
import torch.nn as nn
import torch.nn.functional as func 
import torch.optim as optim
from engine.environment.state import GameState
from engine.environment.observation import ObservationBuilder
import numpy as np
from typing import Tuple, Dict
from engine.agents.policy.normalizer import ValueNormalizer
from engine.agents.policy.swarm_manager import SwarmManager
from engine.utils.enums import AgentRole

GAMMA = 0.95
LAMBDA = 0.95
K_EPOCH = 4
EPS_CLIP = 0.2
ENT_COEF = 0.1


class MAgent:

    def __init__(self, device : torch.device, lr_actor : float = 3e-4, lr_critic : float = 1e-3):

        import os

        print(f"[MAGENT INIT] PID={os.getpid()}")

        print("A")
        self.device = device 
        self.eps_clip = 0.2  ## Epsilon clipping , used at last... for surr 2 that is to limit the actor updates
        self.value_normalizer = ValueNormalizer()

        print("B")

        self.swarm = SwarmManager()

        print("C")
        self.swarm = self.swarm.to(self.device)

    @torch.no_grad()
    def get_actions_and_values(self, obs: np.ndarray, global_state: np.ndarray, action_masks: np.ndarray, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        t_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        t_global_state = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        t_masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)

        dists, values = self.swarm.get_actions_and_values(t_obs, t_global_state, t_masks)

        tank_dist, dealer_dist, healer_dist, boss_dist = dists  ## Shape (num_evns, ) 
        tank_val, dealer_val, healer_val, boss_val = values

        if is_training:
            tank_action = tank_dist.sample()
            healer_action = healer_dist.sample()
            dealer_action = dealer_dist.sample()
            boss_action = boss_dist.sample()            
        else:
            tank_action = torch.argmax(tank_dist.probs, dim = -1)
            healer_action = torch.argmax(healer_dist.probs, dim = -1)
            dealer_action = torch.argmax(dealer_dist.probs, dim = -1)  ## Why are we taking the last dim of that argmax ... We will get the argument where the prob is max but why dim -1
            boss_action = torch.argmax(boss_dist.probs, dim = -1) ## That is because our shape is (num_envs, 8) so we are telling argmax to max over those actions and not along the num_envs * num_agents..

        # Get log probabilities (needed for PPO update)
        tank_log_prob = tank_dist.log_prob(tank_action)
        healer_log_prob = healer_dist.log_prob(healer_action)
        dealer_log_prob = dealer_dist.log_prob(dealer_action)
        boss_log_prob = boss_dist.log_prob(boss_action)

        # 4. RECOMBINE INTO (num_envs, 4) FOR MAIN.PY
        # torch.stack takes our 1D arrays and lines them up as columns!
        actions = torch.stack([tank_action, dealer_action, healer_action, boss_action], dim=1) # Shape (num_envs, num_agents,).. but why dim = 1?? ALong the nd dimension we are saying to stack.. so along columns...
        log_probs = torch.stack([tank_log_prob, dealer_log_prob, healer_log_prob, boss_log_prob], dim=1)
        state_values = torch.stack([tank_val, dealer_val, healer_val, boss_val], dim=1)

        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), state_values.detach().cpu().numpy()


    def update(self, batch : Dict[str, torch.Tensor], ent_coef : float, ppo_epochs : int = 4) -> Dict[str, float]:  # A single update method for a single batch by a single batch creator..

        total_actor_loss = {}
        total_critic_loss = {}

        running_actor = {r : 0.0 for r in AgentRole}
        running_critic = {r : 0.0 for r in AgentRole}  ## These are the sum of losses of individual agents for 4 epochs.. so we take the sum and average out per agent..
        total_entropy = 0.0
        total_count = 0

        self.value_normalizer.update(batch['returns'])  ## Calculate the running mean and all .. we update the normlaizer batch wise and use that for each role normlizing...

        ### Unpacking batch elements.. note that the batch shape is (batch_szie, num_agnets, ..)
        for step in range(ppo_epochs):
            for role in AgentRole:
                b_obs = batch['obs'][:, role, :]              # Shape: (mini_batch_size, 24)
                b_global = batch['global_state'][:, role, :]     # Shape: (mini_batch_size, 96)
                b_actions = batch['actions'][:, role,]        # Shape: (mini_batch_size,)
                b_log_probs = batch['log_probs'][:, role,]    # Shape: (mini_batch_size,)
                b_advs = batch['advantages'][:, role,]         # Shape: (mini_batch_size,)
                b_returns = batch['returns'][:, role,]         # Shape: (mini_batch_size,)  ## There is no need for values here.. we try to bring in critic predicted values as close to returns (advantages + values)
                b_action_masks = batch['action_masks'][:, role, :]   # Shape: (mini_batch_size, 8)
                b_active_masks = batch['active_masks'][:, role,] # Shape: (mini_batch_size,)

                active_sum = torch.clamp(b_active_masks.sum(), min = 1.0)  ## Sum of all the times that single agent was alive...

                ## CRITIC UPDATE 
                #self.value_normalizer.update(b_returns)  ## Calculate the running mean and all
                normalized_returns = self.value_normalizer.normalize(b_returns) ## Shape (batch,)

                pred_values = self.swarm.get_value(b_global, role)  ## Get values for the current role...
                unreduced_critic_loss = func.huber_loss(pred_values, normalized_returns, reduction = 'none')  ## As torch calculates individual losses, it means them and returns a single tensor to us.. we use reduction = None to not mean them and next mean them only for alive agents..
                raw_critic_loss = ((unreduced_critic_loss * b_active_masks) / active_sum).sum() ## .. Calculate loss according to alive agents and such... So this role should get the loss from only when its alive and no uneccesaary garbage needs to be there..

                ### ACTOR UPDATE 
                b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8) ## A small time normalization of the advantages, so that we get small surr 1 and surr 2.. so that those things stabilize..

                dist = self.swarm.get_dist(b_obs, b_action_masks, role)  ## We calculate the action distributions again for the same obs which are already in the buffer
                new_log_probs = dist.log_prob(b_actions) ## And new logs to the same... I guess for the updated network these would change... So we wanted to compare old probs and new probs
                entropy = dist.entropy()
                total_entropy += entropy.mean().item() ## Mean of the whole batch entropies
                total_count += 1
                
                ratios = torch.exp(new_log_probs - b_log_probs)  ## Comparing policy.. that is log probs
                surr1 = ratios * b_advs
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advs ## Only limit the ratio to be somewhat within a range..

                raw_actor_loss = -torch.min(surr1, surr2) - (ent_coef * entropy) ## Maximise both entropy and surr , that is ratio into advantage, that si the new policy should be better htan old and also advantages should be large.. that is how much good is the current action than the past...
                actor_loss = ((raw_actor_loss * b_active_masks) / active_sum).sum()  ## Just sum all those losses divided by the number of times the agent was alive in that batch

                total_actor_loss[role] = actor_loss ##These get overwritten for each epoch and the networks update 4 times with different values orthe same.. based on sampling so
                total_critic_loss[role] = raw_critic_loss

                running_actor[role] += actor_loss.item() / ppo_epochs 
                running_critic[role] += raw_critic_loss.item() / ppo_epochs ## Average out so that we can safely take the mean of 4 epochs and update and return the mean

            self.swarm.update_weights(total_actor_loss, total_critic_loss)

        return{
            "actor_loss" : running_actor,
            "critic_loss" : running_critic,
            "entropy" : total_entropy / total_count, 
            }
            
