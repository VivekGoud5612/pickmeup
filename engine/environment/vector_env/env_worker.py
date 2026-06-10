import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Callable,Any
import traceback


def env_worker(remote : Connection, env_fn : Callable[[], Any]):
    #This fuctions runs in a completely seperate CPU cores
    #It listens to the remote 'pipe' commands,executes them on env and sends results back


    #Instatiate the env inside this isolated process
    #We pass a function 'env_fn' that creates env rather env object
    #Because complex objects often crash when passed across CPU cores
    env = env_fn()

    try:

        while True:
            #Wait for a command from the Main Process
            command, data = remote.recv()

            if command == "step":
                #Data is the action(s) for this environment given by main process from actor network
                obs, reward, done, truncated, info = env.step(data)

                #Automatically reset if the game ends, instead of waiting for the command from main process after every other worker completes
                if done or truncated :
                    #Store the last obs for the agents to see
                    info['terminal_observation'] = obs

                    obs, _ = env.reset()

                #Send the data to the Main process
                remote.send((obs, reward, done, truncated, info))

            elif command == "reset":
                obs, info = env.reset()
                remote.send((obs, info))

            elif command == "close":
                env.close()
                remote.close()
                break   #Break the infinite loop to kill the worker

            else:
                raise NotImplementedError(f"Worker received unknown command: {command}")
            
    except KeyboardInterrupt:
        #Catch Ctrl + C and end the process
        env.close()
        remote.close()

    except Exception as e:
        #Prevent deadlocks by sending the crash report to the main process
        #If worker crashed due to an error,but the main process still waits for remote.recv from this worker
        #So we send the error to the main process,which it can return
        error_trace = traceback.format_exc()
        remote.send(("error", error_trace))
        env.close()
        remote.close()
