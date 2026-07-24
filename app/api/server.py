from fastapi import APIRouter
import asyncio
from app.api.inference import Inference 
import json
import traceback


inference = Inference(checkpoint_path = 'checkpoints/MAPPO_GridWorld_1782322074/step_4800000.pt')
router = APIRouter(prefix = '/api', tags = ['Control'])

#Global State controllers
SIMULATION_RUNNING = False
SIM_TASK = None


#The Background Worker
#This function in the background , completely detached from the HTTP request of the start endpoint.
#Since the main end point returns the success flag, the uvicorn server can take the requests from other user...
#This allows our uvicorn to not wait for the game frames generated..
#The frames will be sent to the node.js backend through redis,,and then to the frontend via websocket connection
async def run_simulation_loop():
    print("[*] Background Task Started")
    global SIMULATION_RUNNING

    try:
        #Max steps
        while SIMULATION_RUNNING:
            #This calculates the frames and automatically send them through redis
            #Thread offloading, Run this in seperate worker thread, since even though the heavy Pytorch is run on GPU,
            #The CPU has to wait for the completion of the frame to send through redis.....
            #This causes a bottleneck in FastApi servers requests..
            #If a user presses start or reset, it takes time for the cpu to resolve frame generation and then serving request
            #For many users..this may take even a second also
            #Only send the callable type to the thread
            frame_data = await asyncio.to_thread(inference.get_next_frame)

            #Auto stop if the episode finishes (Truncated or Terminated)
            if frame_data['step'] >= 200 or inference.info.get('terminal', False):
                print("[*] Episode complete. Auto Pausing Game")
                SIMULATION_RUNNING = False
                break

            #Sleep for 0.1 seconds to get 10fps
            await asyncio.sleep(0.1)
    
        print("[*] Game Over")
    
    except asyncio.CancelledError:
        #This catches the kill signal from the STOP button
        print("[*] Game Paused(Task intercepted)")

    except Exception as e:
        #If pytorch crashes we cna catch adn print for debugging
        print(f"[!] Game crashed in background : {e}")
        traceback.print_exc()
        SIMULATION_RUNNING = False

#The main start api endpoint
@router.post("/start")
async def start_sim():
    global SIMULATION_RUNNING, SIM_TASK

    #If already running, return just the message
    if SIMULATION_RUNNING:
        return {"status": "Game already running"}

    #If it is paused,or to start a fresh game, put the running flag true
    #Start the loop using asyncio, so it runs independently of this HTTP request
    SIMULATION_RUNNING = True
    SIM_TASK = asyncio.create_task(run_simulation_loop())
    return {"status": "Game Started/Resumed"}

#Stop endpoint(PAUSE)
@router.post("/stop")
async def stop_sim():
    global SIMULATION_RUNNING, SIM_TASK

    #Put the flag False
    SIMULATION_RUNNING=False

    #Also terminating the sleeeping task, since when i press start again
    #It is creating a new task, which allows for two taks to run at the same time appending the info at the same place
    #Finally crashing
    if SIM_TASK is not None:
        SIM_TASK.cancel()
        SIM_TASK = None

    return {"status" : "Game Paused"}

#Reset endpoint
@router.post("/reset")
async def reset_sim():
    global SIMULATION_RUNNING, SIM_TASK

    #put the running flag False, to stop the game if running
    SIMULATION_RUNNING = False

    #Also terminating the sleping or completed task if present by any chance
    if SIM_TASK is not None:
        SIM_TASK.cancel()
        SIM_TASK = None

    #Force the environment to reset
    _, _ = inference.env.reset()

    #And send the empty frame through redis to clear the UI
    #Also assign a seperate worker thread
    await asyncio.to_thread(inference.get_next_frame)
    return {"status" : "Game Reset done"}
