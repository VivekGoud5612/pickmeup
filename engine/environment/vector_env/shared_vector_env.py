import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory 
from typing import List, Dict, Callable, Any, Tuple 
import numpy as np 
import sys 
from engine.environment.vector_env.shared_worker import shared_env_worker
from engine.environment.state import GameState
from engine.environment.observation import ObservationBuilder

class SharedSubprocessVectorEnv:

    def __init__(self, env_fns : List[Callable[[], Any]], num_agents : int = 4):
        
        import os

        print(
            f"[VECTOR_ENV CREATED] PID={os.getpid()} id={id(self)}"
        )
        self.closed = False 
        self.num_envs = len(env_fns)
        self.num_agents = num_agents 

# 1. Perfectly aligned shapes (dones is now purely agent-level)
        self.shapes = {
            'obs': (self.num_envs, self.num_agents, ObservationBuilder.OBS_SIZE),
            'global_state': (self.num_envs, self.num_agents, ObservationBuilder.GLOBAL_STATE_SIZE),
            'roles': (self.num_envs, self.num_agents, GameState.NUM_ROLES),  ## Role IDS, for actor role embeddings...
            'rewards': (self.num_envs, self.num_agents),
            'dones': (self.num_envs, self.num_agents),            # Shape matches your buffer exactly
            'action_masks': (self.num_envs, self.num_agents, GameState.NUM_ACTIONS),
            'active_masks': (self.num_envs, self.num_agents),
        }
        
        # Using float32 for dones to match your terminated_array (1.0 or 0.0)
        self.dtypes = {
            'obs': np.float32, 'global_state': np.float32, 'roles': np.int32, 
            'rewards': np.float32, 'dones': np.float32, 
            'action_masks': bool, 'active_masks': bool, 'old_hand' : np.float32,
        }

        self.shms = {}
        self.shm_names = {}

        for key in self.shapes:
            size_in_bytes = int(np.prod(self.shapes[key]) * np.dtype(self.dtypes[key]).itemsize)  ## That np.dtype(dtype).itemsize gives us the bytes occupied by each item inside that array.. Namely size of an integer or float or something else...
            self.shms[key] = SharedMemory(create = True, size = size_in_bytes)  ###WE create here and check for and update in shared worker...
            self.shm_names[key] = self.shms[key].name 
            
        

        self.shared_arrays = {}  ### For the same reason we do this in shared_worker .. To reconstruct the numpy arrays to get the result of the worker into an array and return that.
        for key in self.shapes:  ### This is so that we can send the calculated data upstream to the main file so as let copy that into rollout buffer...

            self.shared_arrays[key] = np.ndarray(self.shapes[key], self.dtypes[key], buffer = self.shms[key].buf)  ## Create an array with the same memory reference as the data in the shared array


            ## Process spawning.. pipes!!
        pipes = [mp.Pipe() for _ in range(self.num_envs)]
        self.remotes, self.worker_remotes = zip(*pipes)  ## *pipes is unpacking that list into tuples of parent child connections...
        self.processes = []

        for rank, (worker_remote, env_fn) in enumerate(zip(self.worker_remotes, env_fns)): ## For each worker remote and env_fn and its index rank
            p = mp.Process(
                target = shared_env_worker,  ## Create a process and sort of assign an instance of that function with our worker remote and env_fn
                args = (worker_remote, env_fn, self.shm_names, self.shapes, self.dtypes, rank),  ## Arguments for single worker, this loop runs and creates processes till the size of nnum envs..
                daemon = True
            )
            self.processes.append(p)  # append all process objects inside that list to use it elsewhere 
            p.start()  ## Start that process.. and it keeps on working till we close it offf....

            
    def reset(self, curriculum_level : int = 1) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        for remote in self.remotes:
            remote.send(("reset", curriculum_level)) ## IF main calls reset, the command is reset and the data is None

        infos = []
        for remote in self.remotes: ## After we sent that, we recieve the data.. the time differnce is usually took care by remote...
            status, payload = remote.recv()
            if status == "error":  ## If the status was error that means if the worker failed to do anything, our rollouts data needs to be perfect, we return an error and stop that
                print(f"[CRITICAL] Worker crashed during reset:\n{payload}")
                self.close()
                raise RuntimeError(payload)  #Instead of sys.exit(1), we can raise this error so that we can catch that..
            infos.append(payload)

        obs_dict = {
            'obs' : self.shared_arrays['obs'].copy(),   ## Just copy the data.. and store that in this dict so as to send that to the main file...
            "roles": self.shared_arrays['roles'].copy(),  ## Of size (num_envs, num_agents, num_roles) ### all the envs data is batched and sent to main throguh this
            "global_state": self.shared_arrays['global_state'].copy()
        }
        
        batched_info = {
            "action_masks": self.shared_arrays['action_masks'].copy(),
            "active_masks": self.shared_arrays['active_masks'].copy(),
        }

        return obs_dict, batched_info


    def step(self, actions : np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:

        for remote, action in zip(self.remotes, actions):  ## Group that specific env action to that specific remote
            remote.send(("step", action))

        infos = []
        truncations = []
        for remote in self.remotes:
            status, payload = remote.recv()
            if status == "error":
                print(f"[CRITICAL] Worker crashed during step:\n{payload}")
                self.close()  ## Call the close function , as to terminate all the pipes and processes.. Because one single instance  failing would be catasrtophic in this system.. 
                raise RuntimeError(payload)  # we need to catch this in main file..
            
            truncated, info = payload
            truncations.append(truncated)  ## As now there are booleans of such (num_envs,) that is for each env we get a boolean
            infos.append(info)  ## Useful

        
        next_obs_dict = {
            'obs' : self.shared_arrays['obs'].copy(),  ##Shape (num_envs, num_agents, 24)
            "roles": self.shared_arrays['roles'].copy(),
            "global_state": self.shared_arrays['global_state'].copy()
        }
        
        rewards = self.shared_arrays['rewards'].copy()
        dones = self.shared_arrays['dones'].copy() # Shape: (num_envs, num_agents)
        
        truncated_array = np.array(truncations, dtype = bool)  ## Convert the array directly to an numpy array

        batched_info = {
            "action_masks": self.shared_arrays['action_masks'].copy(),
            "active_masks": self.shared_arrays['active_masks'].copy(),
            "env_infos": infos # Contains your 'terminal_observation' and 'reward_dict' for MAPPO math, now epsiode rewards and winnin team as well...
        }

        # Clean 5-tuple return format
        return next_obs_dict, rewards, dones, truncated_array, batched_info

    
    def close(self) -> None: ## To close off all pipes, processes and shared memory 
        if self.closed : return 

        for remote in self.remotes:
            try : remote.send(("close", None))  # Order children to close their links to shared memory ..
            except Exception as e: pass 

        for p in self.processes: p.join()  ## For all mp.Processes, join them together into main so that there are extra procesed lying around

        for shm in self.shms.values():
            shm.close() ## Close connection to that shared memory block
            shm.unlink() ## Remove the shared memory altogether 

        self.closed = True ## So that we again dont try to close the closed thing..