import time
import asyncio
from pydantic import BaseModel
from app.api.inference import Inference

DB = {}

#The Base model SessionRequest given by the frontend
class SessionRequest(BaseModel):
    session_id : str


#The GameLoop running in the background, managed by SessionManager, to keep track of session details
#The Session Manager(gameSession) are added to the background tasks, so that FastApi can serve next requests
#The main object run in RAM, and managed by session manager to store the state and remove the session if inactive, to optimize RAM
#Contains methods and variables for these tasks...
class GameSession:
    def __init__(self, checkpoint_path : str):
        self.engine = Inference(checkpoint_path = checkpoint_path)
        
        self.task : asyncio.Task | None = None
        self.is_running = False
        self.last_active = time.time()

