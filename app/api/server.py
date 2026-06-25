import torch
import numpy as np 
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from engine.environment.env import Env 
from engine.utils.enums import AgentID, ActionTypes
from engine.agents.policy.trainer import MAgent  

class Inference:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Booting Inference Engine on {self.device}")

        self.counter = 0
        self.env = Env(grid_size=20, max_steps=200) 
        self.obs, self.info = self.env.reset() 

        self.agents = MAgent(device=self.device)
        self.episode_reward = 0.0 # Track running reward for the episode

        if checkpoint_path:  
            checkpoint = torch.load(checkpoint_path, map_location=self.device) 
            self.agents.swarm.tank_actor.load_state_dict(checkpoint['tank_actor']) 
            self.agents.swarm.dealer_actor.load_state_dict(checkpoint['dealer_actor'])
            self.agents.swarm.healer_actor.load_state_dict(checkpoint['healer_actor'])
            self.agents.swarm.boss_actor.load_state_dict(checkpoint['boss_actor'])
            print(f"[*] Loaded model at checkpoint : {checkpoint_path}")
        else:
            print("[!] Running with randomly initialized actor...")

    def get_next_frame(self):
        current_obs = self.obs['obs']
        current_global = self.obs['global_state_obs']
        dummy_action_mask = np.ones((self.env.num_agents, self.env.state.NUM_ACTIONS), dtype=np.bool_)
        
        with torch.no_grad():
            obs_array = np.expand_dims(current_obs, axis=0) 
            global_array = np.expand_dims(current_global, axis=0)
            action_masks_array = np.expand_dims(dummy_action_mask, axis=0)

            actions, _, _, _ = self.agents.get_actions_and_values( 
                obs=obs_array,
                global_state=global_array,
                action_masks=action_masks_array,
                is_training=False,
            )
            flat_actions = actions[0] 

        self.obs, rewards, dones, truncated, self.info = self.env.step(flat_actions)
        
        # Accumulate reward (assuming heroes reward is at index 0 for simplicity, adjust as needed)
        self.episode_reward += float(rewards[0]) 

        state = self.env.state 
        game_state_payload = {
            "step": self.env.step_count,
            "done": False,
            "episode_reward": self.episode_reward,
            "agents": {
                agent.name: {
                    "x": int(state.positions[agent][0]), 
                    "y": int(state.positions[agent][1]), 
                    "hp": float(state.hp[agent]), 
                    "max_hp": float(state.max_hp[agent]), # fallback
                    "stamina": float(state.stamina[agent]) if hasattr(state, 'stamina') else 0.0,
                    "max_stamina": float(state.max_stamina[agent]) if hasattr(state, 'max_stamina') else 50.0,
                    "cooldowns": {
                        "basic": float(state.cooldowns[agent][0]),
                        "utility": float(state.cooldowns[agent][1]),
                        "ultimate": float(state.cooldowns[agent][2]),
                    },
                    # FIXED: Use agent.value for indexing, .name for JSON string
                    "action": ActionTypes(flat_actions[agent.value]).name,
                } for agent in AgentID
            }
        }

        # Handle Episode End
        if self.info.get('terminal', False) or all(dones) or truncated:
            
            # Determine winner (Custom logic based on your env)
            boss_hp = state.hp[AgentID.BOSS]
            if boss_hp <= 0:
                game_state_payload["winner"] = "HEROES"
            else:
                game_state_payload["winner"] = "BOSS"
                
            game_state_payload["final_return"] = self.episode_reward
            
            if self.counter == 10:
                game_state_payload['done'] = True 
            
            self.counter += 1
            # Reset for next game
            self.obs, self.info = self.env.reset()
            self.episode_reward = 0.0

        return game_state_payload

# --- FASTAPI ROUTER ---
router = APIRouter(prefix='/ws', tags=['Combat'])
inference = Inference(checkpoint_path='checkpoints/MAPPO_GridWorld_1782322074/step_4800000.pt')

@router.websocket("/combat")
async def inference_endpoint(websocket: WebSocket): 
    await websocket.accept() 
    print("[*] Viewer Connected") 

    is_running = False 
    current_speed = 2.0 # Default 2 FPS

    try:
        while True:
            try:
                # Dynamic timeout based on UI speed slider
                wait_time = 1.0 / current_speed if is_running else 0.5
                data = await asyncio.wait_for(websocket.receive_json(), timeout=wait_time)

                if data['command'] == "start":
                    is_running = True
                    current_speed = float(data.get('speed', current_speed))
                    print(f"[*] Game Started at {current_speed} FPS")
                    
                elif data['command'] == "stop":
                    is_running = False 
                    print("[*] Game stopped")
                    
                elif data['command'] == "speed":
                    current_speed = float(data['speed'])
                    print(f"[*] Speed changed to {current_speed} FPS")

            except asyncio.TimeoutError:
                pass # Normal frame tick

            if is_running:
                payload = inference.get_next_frame()
                await websocket.send_text(json.dumps(payload))

                # If the UI requested 1 game and it finished, it will send a stop command next tick
    except WebSocketDisconnect:
        print('[!] Viewer Disconnected...')