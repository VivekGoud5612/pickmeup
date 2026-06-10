import numpy as np
from typing import Dict,Generator,Tuple



class VectorizedRolloutBuffer:
    def __init__(
                self, 
                max_steps : int, 
                num_envs :int, 
                num_agents : int,
                local_obs_dim : int, 
                global_states_dim : int,
                actions_dim : int,
                gamma : float = 0.99, 
                lam : float = 0.95
                ):
        
        #Pre allocate memory grids for N multi environments for faster processing
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.gamma = gamma
        self.lam = lam
        self.pointer = 0  #Pointer to keep track of current step row

        #Pre allocate memory grids for N multi environments for faster processing
        self.local_obs = np.zeros((max_steps, num_envs, num_agents, local_obs_dim), dtype = np.float32)
        self.global_states = np.zeros((max_steps, num_envs, num_agents, global_states_dim), dtype = np.float32)
        self.actions = np.zeros((max_steps, num_envs, num_agents), dtype = np.int64)
        self.masks = np.zeros((max_steps, num_envs, num_agents, actions_dim), dtype = np.float32)
        self.log_probs = np.zeros((max_steps, num_envs, num_agents), dtype = np.float32)
        self.values = np.zeros((max_steps, num_envs, num_agents), dtype = np.float32)
        self.rewards = np.zeros((max_steps, num_envs, num_agents), dtype = np.float32)
        self.dones = np.zeros((max_steps, num_envs, num_agents), dtype = np.float32)
        self.active_masks = np.zeros((max_steps, num_envs, num_agents), dtype = np.float32)

        #Output target array by computed by GAE
        self.returns = np.zeros((max_steps, num_envs, num_agents), dtype = np.float32)

        self.advantages = np.zeros((max_steps, num_envs, num_agents), dtype = np.float32)


    def insert(self, local_obs, global_states, actions, masks, log_probs, values, rewards, dones, active_masks):

        self.local_obs[self.pointer] = np.array(local_obs)
        self.global_states[self.pointer] = np.array(global_states)
        self.actions[self.pointer] = np.array(actions)
        self.masks[self.pointer] = np.array(masks)
        self.log_probs[self.pointer] = np.array(log_probs)
        self.values[self.pointer] = np.array(values)
        self.rewards[self.pointer] = np.array(rewards)
        self.dones[self.pointer] = np.array(dones)
        self.active_masks[self.pointer] = np.array(active_masks)

        self.pointer += 1

    def compute_gae_and_returns(self, next_values, next_dones):

        last_gae = np.zeros((self.num_envs, self.num_agents), dtype = np.float32)

        for step in reversed(range(self.max_steps)):

            if step == self.max_steps - 1:
                next_non_terminal = 1.0 - np.expand_dims(next_dones, axis = -1).astype(np.float32)
                next_val = np.array(next_values, dtype = np.float32)

            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_val = self.values[step + 1]

            #TD Error
            delta = self.rewards[step] + self.gamma * next_non_terminal * next_val - self.values[step]

            #Calculate gae even when dead,so done only when env done
            #So even when the win bounty reward comes ,it knows which action contributed most to the win
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae

            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

        #Reset pointer after max_steps
        self.pointer = 0

    
    def generate_agent_mini_batches(self, agent_id : int, mini_batch_size : int = 256) -> Generator[Tuple[np.ndarray, ...], None, None]:

        total_samples = self.max_steps * self.num_envs

        #Exctract this specific agent's data and flatten out the [max_steps, num_envs] dimensions
        #get the local_obs,global_states..etc for the agent_id and that array is flattened out using reshape.
        #reshape(rows,columns) general representation
        #Here dynamic, reshape(rows,-1) it calculates the total elements/rows and puts in column space.
        l_obs = self.local_obs[:, :, agent_id].reshape(total_samples, -1)
        g_states = self.global_states[:, :, agent_id].reshape(total_samples, -1)

        actions = self.actions[:, :, agent_id].reshape(total_samples)
        masks = self.masks[:, :, agent_id].reshape(total_samples, -1)
        l_probs = self.log_probs[:, :, agent_id].reshape(total_samples)
        returns = self.returns[:, :, agent_id].reshape(total_samples)
        advs = self.advantages[:, :, agent_id].reshape(total_samples)
        active_masks = self.active_masks[:, :, agent_id].reshape(total_samples)

        #Create a 1D array of indices for exctraction of data randomly
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, total_samples, mini_batch_size):
            batch_idx = indices[start_idx : start_idx + mini_batch_size]

            yield(
                l_obs[batch_idx],
                g_states[batch_idx],
                actions[batch_idx],
                masks[batch_idx],
                l_probs[batch_idx],
                returns[batch_idx],
                advs[batch_idx],
                active_masks[batch_idx]
            )