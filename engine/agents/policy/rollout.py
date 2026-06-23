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
        local_role_id_shape : Tuple[int, ...],
        global_state_shape : Tuple[int, ...],
        actions_shape : Tuple[int, ...],
        device : torch.device
    ):
 
        self.num_steps = num_steps
        self.num_envs = num_envs 
        self.num_agents = num_agents
        self.device = device 
        self.pointer = 0

        self.obs = np.zeros((num_steps, num_envs, num_agents) + (obs_shape,), dtype = np.float32)
        self.role_ids = np.zeros((num_steps, num_envs, num_agents) + (local_role_id_shape,), dtype = np.int32)
        self.state = np.zeros((num_steps, num_envs, num_agents) + (global_state_shape,), dtype = np.float32)  ## Now each agent as well.. Simply because it would be easy .., we could store it and use it by repeating as before but env returning (num_envs, num_agents, 96) would be more easy to implement
        self.actions = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.rewards = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.values = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)  ## Again for number_of_agents, as we will get different values for each agent...
        self.log_probs = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.advantages = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.returns = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)

        self.dones = np.zeros((num_steps, num_envs, num_agents), dtype = np.float32)
        self.active_masks = np.zeros((num_steps, num_envs, num_agents), dtype = np.bool_)
        self.action_masks = np.zeros((num_steps, num_envs, num_agents) + (actions_shape,), dtype = np.bool_)

    def store(
        self,
        local_obs : np.ndarray,
        local_ids : np.ndarray,
        global_state : np.ndarray,
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
        self.role_ids[self.pointer] = local_ids
        self.state[self.pointer] = global_state
        self.actions[self.pointer] = actions
        self.log_probs[self.pointer] = log_probs
        self.rewards[self.pointer] = rewards
        self.dones[self.pointer] = dones
        self.values[self.pointer] = values
        self.action_masks[self.pointer] = action_masks
        self.active_masks[self.pointer] = active_masks

        self.pointer += 1

    ##  Actually what I wanted to do.. was keep one state, one value for each environment. ANd returns and advantages are different for each agent as both are dependent on step reward right? And each agent has different reward. So lets just change value and expand that. And in gae calculation, I need to check value in TD error. So I need to expand values there as well... 
    def compute_returns_and_advantages(self, next_values : np.ndarray, next_dones : np.ndarray, gamma : float = 0.99, gae_lambda = 0.95):

        last_gae = np.zeros((self.num_envs, self.num_agents), dtype = np.float32)  # No need for step dim as we calculate last_gae of delta for each step

        for step in reversed(range(self.num_steps)):   # Next value of size (batch_size, 1).. hoping num_steps = batch_size... but we should pass next value and done to be of size (num_envs).. lets see about this in 
            
            if step == self.num_steps - 1: ## Check if this is the last step. There is no next value so we calculate that and send it as argumetns to this function.
                next_non_terminal = (1.0 - next_dones.astype(np.float32))  ## next_non_terminal also has the same shape as next_dones (num_envs, num_agents)    
                next_value = next_values 

            else:
                next_non_terminal = (1.0 - self.dones[step + 1].astype(np.float32))  #Terminal condition for each env. Then we multiply it with the future term becasue if that this 0, then there is no meaning calculate the future value
                next_value = self.values[step + 1]

        ## No need for expansion now.. as all our shapes are aligned...
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]  # Shape (num_envs, num_agents)
    
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae #* self.active_masks[step] #shape (num_envs, num_agents)
            self.advantages[step] = last_gae  ## The above next non terminal is the one checking if the agent is done (dead or not and if yes.. It stops GAE next value thing and just focuses on current value)

        self.returns = self.advantages + self.values  # Need to expand self.values before calculating returns , #Shape (num_envs, num_agents)  
        ## Moved advantage mean and std to trainer ... That is advantage normalization..

    def generate_batch(self, batch_size : int) -> Generator[Dict[str, torch.Tensor], None, None]:

        flat_size = self.num_steps * self.num_envs * self.num_agents ## There is no need for us to go agent by agent, due to the coming of role embeddings, it has become much easier for the network to know which agent is currently running and such...

        indices = np.arange(flat_size)
        np.random.shuffle(indices)

        obs_tensor = torch.as_tensor(self.obs.reshape(flat_size, -1), device = self.device)
        role_ids_tensor = torch.as_tensor(self.role_ids.reshape(flat_size, -1), device = self.device)
        actions_tensor = torch.as_tensor(self.actions.reshape(flat_size, -1), device = self.device)
        log_probs_tensor = torch.as_tensor(self.log_probs.reshape(flat_size), device = self.device)
        advantages_tensor = torch.as_tensor(self.advantages.reshape(flat_size), device = self.device)
        action_masks_tensor = torch.as_tensor(self.action_masks.reshape(flat_size, -1), device = self.device)
        active_masks_tensor = torch.as_tensor(self.active_masks.reshape(flat_size), device=self.device)

        state_tensor = torch.as_tensor(self.state.reshape(flat_size, -1), device = self.device)   # * is the unpacking operator, we unpack that tensor.. more detailed below
        values_tensor = torch.as_tensor(self.values.reshape(flat_size), device = self.device)   # *(something) is used for unpacking that tuple into comma seperated values. This way I can unpack state dimension (global state dimension to a comma seperated dim value).
        returns_tensor = torch.as_tensor(self.returns.reshape(flat_size), device = self.device)
    
        for start_idx in range(0, flat_size, batch_size):
            end_idx = start_idx + batch_size 
            mb_indices = indices[start_idx : end_idx]

            yield {
                
                "obs": obs_tensor[mb_indices],
                "role_ids" : role_ids_tensor[mb_indices],
                "actions": actions_tensor[mb_indices],
                "log_probs": log_probs_tensor[mb_indices],
                "advantages": advantages_tensor[mb_indices],
                'action_masks' : action_masks_tensor[mb_indices],
                "active_masks": active_masks_tensor[mb_indices],

                "global_state": state_tensor[mb_indices], ## I guess values uneccesary as we compare loss to returns which values + advantages..
                "returns": returns_tensor[mb_indices],
            }


    def generate_critic_batch(self, batch_size : int) -> Generator[Dict[str, torch.Tensor], None, None]:

        flat_size = self.num_steps * self.num_envs * self.num_agents
        
        indices = np.arange(flat_size)
        np.random.shuffle(indices)

        expanded_state = np.expand_dims(self.state, axis = 2)
        repeated_state = np.repeat(expanded_state, self.num_agents, axis = 2)

        expanded_values = np.expand_dims(self.values, axis = 2)
        repeated_values = np.repeat(expanded_values, self.num_agents, axis = 2)  # But auto broadcasts does not work for reshape operations..

        state_tensor = torch.as_tensor(repeated_state.reshape(flat_size, *self.state.shape[2:]), device = self.device)   # * is the unpacking operator, we unpack that tensor.. more detailed below
        values_tensor = torch.as_tensor(repeated_values.reshape(flat_size), device = self.device)   # *(something) is used for unpacking that tuple into comma seperated values. This way I can unpack state dimension (global state dimension to a comma seperated dim value).
        returns_tensor = torch.as_tensor(self.returns.reshape(flat_size), device = self.device)

        expanded_global_ids = self.global_ids.unsqueeze(0).expand(flat_size, -1)   ## Here we are simply adding a new dim at 0th index and using expand to stretch the tensor along that new dimension with batch size. The -1 is so that the last dim is safe and letting it be whatever it was..

        for start_idx in range(0, flat_size, batch_size):
            end_idx = start_idx + batch_size 
            mb_indices = indices[start_idx : end_idx]
        
            yield {

                "global_state": state_tensor[mb_indices],
                "global_ids" : expanded_global_ids[mb_indices],
                "values": values_tensor[mb_indices],
                "returns": returns_tensor[mb_indices],
            }


    def generate_actor_batch(self, agent_id : int, batch_size : int) -> Generator[Dict[str, torch.Tensor], None, None] :

        flat_envs = self.num_steps * self.num_envs

        indices = np.arange(flat_envs)
        np.random.shuffle(indices)

        agent_obs = self.obs[:, :, agent_id]
        agent_role_ids = self.role_ids[:, :, agent_id]
        agent_actions = self.actions[:, :, agent_id]
        agent_log_probs = self.log_probs[:, :, agent_id]
        agent_advantages = self.advantages[:, :, agent_id]
        agent_action_masks = self.action_masks[:, :, agent_id]
        agent_active_masks = self.active_masks[:, :, agent_id]

        obs_tensor = torch.as_tensor(agent_obs.reshape(flat_envs, *agent_obs.shape[2:]), device = self.device)
        local_ids_tensor = torch.as_tensor(agent_role_ids.reshape(flat_envs, *agent_role_ids.shape[2:]), device = self.device)
        actions_tensor = torch.as_tensor(agent_actions.reshape(flat_envs, *agent_actions.shape[2:]), device = self.device)
        log_probs_tensor = torch.as_tensor(agent_log_probs.reshape(flat_envs), device = self.device)
        advantages_tensor = torch.as_tensor(agent_advantages.reshape(flat_envs), device = self.device)
        action_masks_tensor = torch.as_tensor(agent_action_masks.reshape(flat_envs), device = self.device)
        active_masks_tensor = torch.as_tensor(agent_active_masks.reshape(flat_envs), device=self.device)

        for start_idx in range(0, flat_envs, batch_size):
            end_idx = start_idx + batch_size 
            mb_indices = indices[start_idx : end_idx]

            yield {
                "obs": obs_tensor[mb_indices],
                "local_ids" : local_ids_tensor[mb_indices],
                "actions": actions_tensor[mb_indices],
                "log_probs": log_probs_tensor[mb_indices],
                "advantages": advantages_tensor[mb_indices],
                'action_masks' : action_masks_tensor[mb_indices],
                "active_masks": active_masks_tensor[mb_indices],
            }

    def clear(self):
        self.pointer = 0