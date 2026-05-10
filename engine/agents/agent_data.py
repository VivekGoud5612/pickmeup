from .data import CombatStats
from dataclasses import dataclass

@dataclass
class AgentIdentityFormat:
    id : int
    pos : tuple[int, int]
    role: str
    stats : CombatStats

class AgentIdentity:
    def create_identity(self, agent_id : int, role: str = None):  ## For now , later we use this to create diverse agent identities and base agent takes that and 
         ## the rule based or LLM agent with some sort of identity
        
        _data = _get_data(role)
        return AgentIdentityFormat(
            id = agent_id,
            pos = _data['pos'],
            role = role,
            combat_stats = CombatStats(hp = _data['hp'], attack = _data['attack'], min_range = _data['min_range'], max_range = _data['max_range'] )
        )
    
    @staticmethod
    def _get_data(role : str):
        if role == "Tank":
            return {
                "pos": (0, 0),
                "hp": 150,
                "attack": 10,
                "min_range": 1,
                "max_range": 1
            }

        elif role == "Dealer":
            return {
                "pos": (0, 1),
                "hp": 80,
                "attack": 25,
                "min_range": 2,
                "max_range": 4
            }

        elif role == "Boss":
            return {
                "pos": (9, 9),
                "hp": 300,
                "attack": 20,
                "min_range": 1,
                "max_range": 2
            }

        return {}
