from fastapi import APIRouter, BackgroundTasks
import uvicorn 
import asyncio
from app.api.inference import Inference 
import json


inference = Inference(checkpoint_path = 'checkpoints/MAPPO_GridWorld_1781890067/step_4800000.pt')
router = APIRouter(prefix = '/api', tags = ['Control'])

#The Background Worker
#Thsi function in the background , completely detached from the HTTP request of the start endpoint.
#Since the main end point returns the success flag, the uvicorn server can take the requests from other user...
#This allows our uvicorn to not wait for the game frames generated..
#The frames will be sent to the node.js backend through redis,,and then to the frontend via websocket connection
async def run_game_loop():
    print("[*] Background Task Started")

    #Max steps
    for _ in range(200):
        #This calculates the frames and automatically send them through redis
        inference.get_next_frame()

        #Sleep for 0.1 seconds to get 10fps
        await asyncio.sleep(0.1)
    
    print("[*] Game Over")


#The main api endpoint
@router.post("/start")
async def start_simulation(background_taks : BackgroundTasks):
    #Give the gaem loop to the fastapi's background thread
    background_taks.add_task(run_game_loop)

    #Return ok to the frontend
    return{
        "status" : "success",
        "message" : "Broadcasting on redis port 6379"
    }
