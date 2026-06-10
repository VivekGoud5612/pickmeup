from .state import GameState 
from .reward import Reward_Calculator
from engine.actions.action_sequencer import Action_Sequencer
from typing import List, Dict, Any
import copy


class RaidEnv:
    def __init__(self, grid_size : int = 20, max_steps : int = 200):

        self.grid_size = grid_size
        self.max_steps = max_steps      #To update the truncated flag

        self.gamestate = None
        self.reward_calc = Reward_Calculator()
        self.step_count = 0

        #We no longer instantiate Agent/policy objects here
        #Instead the environment only needs to know the Id's and roles
        self.agents = {
            0 : "Tank",
            1 : "Dealer",
            2 : "Healer",
            3 : "Boss"
        }
        