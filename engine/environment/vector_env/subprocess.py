import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Dict, Callable, Any, Tuple, List
import numpy as np
from engine.environment.vector_env.env_worker import env_worker
import sys


class SubprocessVectorEnv:
    #The main process.It manages multiple paraller environment workers running on seperate CPU cores 
    #using Pipes
    def __init__(self, env_fns : List[Callable[[], Any]]):
        self.closed = False
        self.num_envs = len(env_fns)

        #mp.Pipe returns a Tuple : (manager_end, worker_end).
        #Create pipes for every env_worker and manager
        pipes = [mp.Pipe() for _ in range(self.num_envs)]

        #Now *pipes unpacks the list pipes.
        #Zip() returns the first pile(manager_end-1,manager_end-2....) and second pile(worker_end-1,worker_end-2...)
        #Since manager end pipe should be with manager 
        #And worker end pipe should be distributed to every worker
        self.remotes, self.work_remotes = zip(*pipes)

        self.processes = []   #List of processes
        
        #Matches the work_remote to its respective env_fn
        #Like worker remote -1 to env_fn - 1
        for rank, (work_remote, env_fn) in enumerate(zip(self.work_remotes, env_fns)):       #rank is used to create a different env if evey env_fn creates the same environment
            p = mp.Process(
                target = env_worker,                   #env_worker works in that process
                args = (work_remote, env_fn),          #That worker uses its respective remote(pipe) and env_fn
                daemon = True                          #Dead Man's Switch, worker dies ,if main.py crashes
            )
            self.processes.append(p)                   #Store the process in the list
            p.start()                                  #Start the process

        #Close the worker side pipe in the main process
        #Only workers shoudl hold their side open
        for work_remote in self.work_remotes:
            work_remote.close()


    def step(self, actions : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        #Broadcasts actions to all the parallel envs,wait for them to step.
        #and packages the result into clean arrays for the buffer

        #Send actions to env_workers by self.remote one by one
        #Ex : action-1 to env-worker-1 usign self.remote(maganer_end-1)
        #Match actions to respective remotes
        for remote,action in zip(self.remotes, actions):
            remote.send(('step', action))

        #Wait and collect the responses from the workers
        results = [remote.recv() for remote in self.remotes]

        #Check if any worker sent an error crash
        for result in results:
            if isinstance(result, tuple) and result[0] == "error":
                print("[CRITICAL] One of the workers crashed due to error:")
                print(result[1])
                self.close()
                sys.exit(1)

        #Unpack the results: reults looks like [(obs, rewards, dones, truncateds, infos), (obs, rewards, dones, truncateds, infos)...]
        #Zip() returns all(obs), all(rewards), all(dones), all(truncateds), all(infos)
        obs, rewards, dones, truncateds, infos = zip(*results)

        #Stack them into neat, high-speed Numpy arrays for your Buffer
        return np.stack(obs), np.array(rewards, dtype = np.float32), np.array(dones, dtype = np.float32), np.array(truncateds, dtype = np.float32), list(infos)
    

    def reset(self) -> np.ndarray:
        #Resets all parallel environments and returns their starting screens
        for remote in self.remotes:
            remote.send(('reset', None))

        results = [remote.recv() for remote in self.remotes]

        #Check if any worker sent an error crash
        for result in results:
            if isinstance(result, tuple) and result[0] == "error":
                print("[CRITICAL] One of the workers crashed due to error:")
                print(result[1])
                self.close()
                sys.exit(1)

        obs, infos = zip(*results)

        return np.stack(obs)
    

    def close(self):
        #Shuts down all paralle environments and cleans up memory
        if self.closed:
            return
        
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except Exception:
                pass                             #If worker is already dead,ignore

        #Wait for all processes to close
        for p in self.processes:
            p.join()
        
        self.closed = True
