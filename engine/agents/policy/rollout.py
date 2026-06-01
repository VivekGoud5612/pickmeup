import numpy as np
import torch
from typing import Generator, Dict, Tuple
class RolloutBuffer:

    def __init__(
        self,
        num_steps : int,
        num_envs : int,
        num_agents : int,
        obs_shape : Tuple[int, ...],
        global_state_shape : Tuple[int, ...],
        action_shape : Tuple[int, ...],
        device : torch.device
    ):
 
        self.num_steps = num_steps
        self.num_envs = num_ens 
        self.num_agents = num_agents
        self.device = device 
        self.pointer = 0

        self.obs = np.zeros((num_steps, num_envs, num_agents) + obs_shape, dtype = np.float32)
        self.state = np.zeros((num_steps, num_envs) + global_state_shape, dtype = np.float32)
        self.actions = np.zeros((num_steps, num_envs, num_agents) + action_shape, dtype = np.float32)
        self.rewards = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype = np.float32)
        self.log_probs = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.advantages = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.returns = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)

        self.dones = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.active_masks = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.action_masks = np.zeros((num_steps, num_ens, num_agents) + action_shape, dtype = np.float32)

    def store(
        self,
        local_obs : np.ndarray,
        global_states : np.ndarray,
        actions : np.ndarray,
        log_probs : np.ndarray,
        rewards : np.ndarray,
        dones : np.ndarray,
        values : np.ndarray,
        action_masks : np.ndarray,
        active_masks : np.ndarray,
    ):

        assert self.pointer < self.num_steps, 'Rollout buffer overflow. Call compute_returns() and clear.'

        self.obs[self.pointer] = local_obs
        self.global_states[self.pointer] = global_states
        self.actions[self.pointer] = actions
        self.log_probs[self.pointer] = log_probs
        self.rewards[self.pointer] = rewards
        self.dones[self.pointer] = dones
        self.values[self.pointer] = values
        self.action_masks[self.pointer] = action_masks
        self.active_masks[self.pointer] = active_masks

        self.pointer += 1

    ##  Actually what I wanted to do.. was keep one state, one value for each environemtn. ANd returns and advantages are different for each agent as both are dependent on step reward right? And each agent has different reward. So lets just change value and expand that. And in gae calculation, I need to check value in TD error. So I need to expand values there as well... 
    def compute_returns_and_advantages(self, next_values : np.ndarray, next_dones : np.ndarray, gamma : float = 0.99, gae_lambda = 0.95):

        last_gae = np.zeros((self.num_envs, self.num_agents), dtype = np.float32)

        for step in reversed(range(self.num_steps)):
            
            current_expanded_value = np.expand_dims(self.values[step], axis = 1)

            if step == self.num_steps - 1: ## Check if this is teh last step. There is no next value so we calculate that and send it as argumetns to this function.
                next_non_terminal = (1.0 - next_dones.astype(np.float32))    

                next_value = np.expand_dims(next_values, axis = 1) 

            else:
                next_non_terminal = (1.0 - self.dones[step + 1].astype(np.float32))  #Terminal condition for each env. Then we multiply it with the future term becasue if tha this 0, then there is no meaning calculate the future value

                next_value = np.expand_dims(self.values[step + 1], axis = 1)


            delta = self.rewards[step] + gamma * next_value * next_non_terminal - current_expanded_value
            delta = delta * self.active_masks[step]
    
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae * self.active_masks[step]
            self.advantages[step] = last_gae 

        expanded_values = np.expand_dims(self.value, axis = 2)   # converts [num_envs,] -> [num_envs, 1]  So np automatically broadcasts the same from [num_envs, 1] -> [num_envs, num_agents]. Because numpy automatically broadcasts or stretches with operations like (+, -, *).
        self.returns = self.advantages + expanded_values  # Need to expand self.values before calculating returns 

        advantages_mean = self.advantages.mean()
        advantages_std = self.advantages.std()

        self.advantages = (self.advantages - advantages_mean) / (advantages_std + 1e-8)


    def generate_critic_batch(self, batch_size : int) -> Generator[Dict[str, torch.Tensor], None, None]:

        flat_size = self.num_steps * self.num_envs * self.num_agents
        
        indices = np.arange(flat_size)
        np.random.shuffle(indices)

        expanded_state = np.expand_dims(self.state, axis = 2)
        repeated_state = np.repeat(expanded_state, self.num_agents, axis = 2)

        expanded_values = np.expand_dims(self.values, axis = 2)
        repeated_values = np.repeat(expanded_values, self.num_agents, axis = 2)  # But auto broadcasts does not work for reshape operations..

        state_tensor = torch.as_tensor(repeated_state.reshape(flat_size, *self.state.shape[2:]), device = self.device)
        values_tensor = torch.as_tensor(repeated_values.reshape(flat_size), device = self.device)   # *(something) is used for unpacking that tuple into comma seperated values. This way I can unpack state dimension (global state dimension to a comma seperated dim value).
        returns_tensor = torch.as_tensor(self.returns.reshape(flat_size), device = self.device)

        for start_idx in range(0, flat_size, batch_size):
            end_idx = start_idx + batch_size 
            mb_indices = indices[start_idx : end_idx]

            yield {

                "global_state": state_tensor[mb_indices], 
                "values": values_tensor[mb_indices],
                "returns": returns_tensor[mb_indices],
            }


    def generate_actor_batch(self, agent_id : int, batch_size : int) -> Generator[Dict[str, torch.Tensor], None, None] :

        flat_envs = self.num_steps * self.num_envs

        indices = np.arange(flat_envs)
        np.random.shuffle(indices)

        agent_obs = self.obs[:, :, agent_id]
        agent_actions = self.actions[:, :, agent_id]
        agent_log_probs = self.log_probs[:, :, agent_id]
        agent_advantages = self.advantages[:, :, agent_id]
        agent_action_masks = self.action_masks[:, :, agent_id]
        agent_active_masks = self.active_masks[:, :, agent_id]

        obs_tensor = torch.as_tensor(agent_obs.reshape(flat_envs, *agent_obs.shape[2:]), device=self.device)
        actions_tensor = torch.as_tensor(agent_actions.reshape(flat_envs, *agent_actions.shape[2:]), device=self.device)
        log_probs_tensor = torch.as_tensor(agent_log_probs.reshape(flat_envs), device=self.device)
        advantages_tensor = torch.as_tensor(agent_advantages.reshape(flat_envs), device=self.device)
        action_masks_tensor = torch.as_tensor(agent_action_masks.reshape(flat_envs))
        active_masks_tensor = torch.as_tensor(agent_active_masks.reshape(flat_envs), device=self.device)

        for start_idx in range(0, flat_envs, batch_size):
            end_idx = start_idx + batch_size 
            mb_indices = indices[start_idx : end_idx]

            yield {
                "obs": obs_tensor[mb_indices],
                "actions": actions_tensor[mb_indices],
                "log_probs": log_probs_tensor[mb_indices],
                "advantages": advantages_tensor[mb_indices],
                'action_masks' : action_masks_tensor[mb_indices],
                "active_masks": active_masks_tensor[mb_indices],
            }

    def clear(self):
        self.pointer = 0