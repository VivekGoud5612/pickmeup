from abc import abstractmethod, ABC 
from .engine.environment import GameState

class BasePolicy(ABC):
    
    @abstractmethod
    def decide_action(self, agent_id: int, state:GameState):
        raise NotImplementedError