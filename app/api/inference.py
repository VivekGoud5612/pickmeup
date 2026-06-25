import torch
import numpy as np 
from engine.environment.env import Env 
from engine.environment.state import GameState 
from engine.environment.observation import ObservationBuilder
from engine.utils.enums import AgentID, ActionTypes
from engine.agents.policy.trainer import MAgent  


class Inference:

    def __init__(self, checkpoint_path = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*]Booting Inference Engine on {self.device}")

        self.env = Env(grid_size = 20, max_steps = 200)  ## Here we just initialize a single env because this is not trainingand there is no need for those many envs, there isonly one env
        self.obs, info = self.env.reset()  ## Reset the env during this class initialization so that the step can go on in a loop

        self.agents = MAgent(device = self.device)

        if checkpoint_path:  ## IF the checkpoint path is not None, that is if there is a saved model then 
            checkpoint = torch.load(checkpoint_path, map_location = self.device) ## load the weihts from the path onto the device
            self.agents.swarm.tank_actor.load_state_dict(checkpoint['tank_actor'])  ## WE load the actor weights because right now there is no need for critic, as the actor is already trained.. and we need only actions
            self.agents.swarm.dealer_actor.load_state_dict(checkpoint['dealer_actor'])
            self.agents.swarm.healer_actor.load_state_dict(checkpoint['healer_actor'])
            self.agents.swarm.boss_actor.load_state_dict(checkpoint['boss_actor'])

            print(f"[*] We loaded the saved model at checkpoint : {checkpoint_path}")

        else:
            print("[!] Running with randomly initialized actor...")


    def get_next_frame(self): ## That is a function for the websocket endpoint to get the next state after a step

        current_obs = self.obs['obs']
        current_global = self.obs['global_state_obs']
        
        dummy_action_mask = np.ones((self.env.num_agents, self.env.state.NUM_ACTIONS), dtype = np.bool_)
        
        with torch.no_grad():
             ## Add batch dim to these arrays  .. as the network is tuned to work on 3D Data...

            obs_array = np.expand_dims(current_obs, axis = 0)  ## Shape (1, 4, 24)
            global_array = np.expand_dims(current_global, axis = 0)
            action_masks_array = np.expand_dims(dummy_action_mask, axis = 0)

            actions, _, _, _ = self.agents.get_actions_and_values(  ## Shape (1,4)
                obs = obs_array,
                global_state = global_array,
                action_masks = action_masks_array,
                is_training = False,
            )

            flat_actions = actions[0] ## Take the first line so shape is (4,)

        self.obs, _, dones, truncated, self.info = self.env.step(flat_actions)

        state = self.env.state  ## Store the state class after step.. so that we can send hte JSON to frontend via router..3.
        game_state_payload = {
            "step": self.env.step_count,
            "done"  : False,
            "agents": {
                    agent.name : {
                        "x": int(state.positions[agent][0]), 
                        "y": int(state.positions[agent][1]), 
                        "hp": float(state.hp[agent]), 
                        "max_hp": float(state.max_hp[agent]),
                        'stamina' : float(state.stamina[agent]),
                        'max_stamina' : float(state.max_stamina[agent]),
                        'cooldowns': {
                            "basic" : float(state.cooldowns[agent][0]),
                            "utility" : float(state.cooldowns[agent][1]),
                            "ultimate" : float(state.cooldowns[agent][2]),
                        },
                        'action' : ActionTypes(flat_actions[agent]),
                    }  for agent in AgentID
            },
            "win_rate" : 0.0,
        }

        if self.info['terminal'] or all(dones) or truncated:
            game_state_payload['win_rate'] = self.info['win_rate']
            self.obs, self.info = self.env.reset()
            game_state_payload["done"] = True

        return game_state_payload