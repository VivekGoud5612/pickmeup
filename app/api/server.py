from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import uvicorn 
import asyncio
from app.api.inference import Inference 
import json


inference = Inference(checkpoint_path = 'checkpoints/MAPPO_GridWorld_1782322074/step_4800000.pt')
router = APIRouter(prefix = '/ws', tags = ['Combat'])

@router.websocket("/combat")
async def inference_endpoint(websocket : WebSocket):  ## A websocket endpoint for the frontend to access
    await websocket.accept()   ## Awaiting a conection of a websocket connection to this endpoint
    print("[*]Viewer Connected")   ## If the await connection ran then we check via this message

    try:
        while True:  ## Now send the data to browser repeatedly...
            game_state_payload = inference.get_next_frame()  ## Get next frame and send it via web socket
            await websocket.send_text(json.dumps(game_state_payload)) # Awaiting the control back here whenever the data is accepted at the receivers end
            await asyncio.sleep(0.5)  ## Wait 0.1 sec for each frame so that the game runs at 10FPS (so that we cna see it clearly)

    except WebSocketDisconnect:
        print('[!] Viewer Disconnected...')

