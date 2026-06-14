import traceback
import numpy as np 
from multiprocessing.connection import Connection 
from multiprocessing import shared_memory 
from typing import Callable, Any 

def shared_env_worker(remote : Connection, env_fn : Callable[[], Any], shm_names : np.ndarray, shapes : np.ndarray, dtypes : np.ndarray, env_idx : int):  # remote is a connection, others are in notes and Callable is a type hint we use for functions, where Callable[[for arguments], [for returns]]
## Also shapes is also used for the Numpy object reconstruction
    obs_shm = shared_memory.SharedMemory(name = shm_names[])   ## Get the local views or you could say raw bytes from the memory (shared)
    state_shm = shared_memory.SharedMemory(name = shm_names['state'])
    roles_shm = shared_memory.SharedMemory(name = shm_names['roles'])
    rewards_shm = shared_memory.SharedMemory(name = shm_names['rewards'])
    dones_shm = shared_memory.SharedMemory(name = shm_names['dones'])
    action_masks_shm = shared_memory.SharedMemory(name = shm_names['action_masks'])  ## Contains both action and active masks.. but I changed that
    active_masks_shm = shared_memory.SharedMemory(name = shm_names['active_masks'])

    global_obs = np.ndarray(shapes['obs'], dtype = dtypes['obs'], buffer = obs_shm.buf)[env_idx] ### Reconstruct the numpy object with that shape, data type, and with buffer it knows what memory to point to take in data.. And we slice the data for this single environment
    global_roles = np.ndarray(shapes['roles'], dtype = dtypes['roles'], buffer = roles_shm.buf)[env_idx] ## Notice that here we
    global_rewards = np.ndarray(shapes['rewards'], dtype = dtypes['rewards'], buffer = roles)[env_idx]
    global_dones = np.ndarray(shapes['dones'], dtype = dtypes['dones'], buffer = dones_shm.buf)[env_idx]
    global_action_masks = np.ndarray(shapes['action_masks'], dtype = dtypes['action_masks'], buffer = action_masks_shm.buf)[env_idx]
    global_active_masks = np.ndarray(shapes['active_masks'], dtype = dtypes['active_masks'], buffer = active_masks_shm.buf)[env_idx]

    env = env_fun()  ## Instantiate the local environment before using it in the process elsewhere

    try:
        while True:   
        ## Low bandwidth command synchronization via Pipe
        command, date = remote.recv()  ## This command and data are sent via Pipe from main process, and as the conversion of these small elements to bytes and to elements from one end to another end of the 

        if command == "step":
            # data corresponds to actions array slice of shape (num_agents,)
            obs, reward, done, truncated, info = env.step(data)  ## These are the things returned by the environment - Standard gym format, truncated means if something like the buffer size is reached so it stops the env step process and we return True

            if done or truncated:
                info['terminal_observation'] = obs.copy() ## To store the very last observation of this step (After the whole required number of steps run or the episode completes)
                obs, _ = env.reset() 


            global_obs[:] = obs['obs']  # We copy the elements into global obs as to avoid referencing to the same memory as obs['obs]
            global_roles[:] = obs['roles']
            global_roles[:] = reward 
            global_dones[:] = done 
            global_action_masks = info['action_masks']
            global_action_masks = info['active_masks']


            ## Notifying the main process that the execution is complete, we pass the command ok and metadata like truncated and info along the pipe...
            remote.send(("ok", (truncated, info)))


        if command == "reset":

            obs, info = env.reset()  ## Need to write what env.reset() returns
            global_obs[:] = obs["obs"]
            global_roles[:] = obs['roles']
            global_action_masks[:] = info['action_masks']  ## IS active mask important in reset ??
            global_active_masks[:] = info['active_masks']  
            remote.send(('ok', info))

        
        elif command == "close":  ## Close this shared worker thing and release connections to that memory (that is close shm connections)
            break 

        else:
            raise NotImplementedError(f"Worker received invalid command: {command}")

    except KeyboardInterrupt:
        pass 
    except Exception as e:
        error_trace = traceback.format_exc() ## Trace where the error occured and why the except block ran
        remote.send(("error", error_trace))  ## Send this error to the main process so that we can reason on how to resolve this error

    finally : ## Doesn't matter if try or except ran.. we finally run this to close the memory attachments (all shms)
        for shm in [obs_shm, roles_shm, rewards_shm, dones_shm, action_masks_shm, active_masks_shm]:
            shm.close() ## Close the connection...
        remote.close()  ## Close the pipe as well...
        ## We close all of these finally after the training is done or the close is called ... In the while loop above, the only two ways to close this connection is when close() is called or try fails