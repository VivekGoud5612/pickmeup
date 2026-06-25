from engine.agents.agent_data import AgentIdentity, AgentRole
from engine.agents.policy.trainer import MAgent
from typing import List
from engine.environment.observation import Observation
from engine.actions.action import ActionTypes
import numpy as np
from engine.environment.state import GameState 
from engine.environment.env import AgentID

class BaseAgent:

    def __init__(self, agent_id : AgentID, role : AgentRole):  ## We create 4 objects of this base agent in init of env
      
        self.identity = AgentIdentity.create_identity(agent_id, role)  # Here we create a pure object of the agent, and the copy the information to state as pure arrays..
        self.agent = MAgent(agent_id, role)

    def get_action_and_log_probs(self, state : GameState, observation : np.ndarray, role_ids : np.ndarray, action_mask : np.ndarray, is_training : bool):
        
        action_idx, log_probs = self.agent.get_action(observation, role_ids, action_mask, is_training)

        action_idx = action_idx.item()  ## we take the item of a tensor and it converts that to a simple integer
        log_probs_numpy_array = log_probs.numpy()  ## Numpy method of tensor

        if action_idx in ActionTypes:
            action_type_enum = ActionTypes(action_idx)  ## If valid action - action type enum is equal to the number
        
        else:
            action_type_enum = ActionTypes.WAIT 
            state.invalid_actions[AgentID] = True 

        return action_type_enum, log_probs 


            