from dataclasses import dataclass
import random

@dataclass
class Skill:
    power:int
    min_range:int
    max_range:int
    cooldown:int

@dataclass
class CombatStats:
    hp:int
    max_hp:int
    skills:dict[str,Skill]

@dataclass
class AgentIdentityFormat:
    id : int
    pos : tuple[int, int]
    role: str
    stats : CombatStats
    action_space_size:int=7

class AgentIdentity:
    def create_identity(self, agent_id : int, role: str = None, grid_size: int =20):  ## For now , later we use this to create diverse agent identities and base agent takes that and 
         ## the rule based or LLM agent with some sort of identity
        
        _data = self._get_data(role,grid_size)
        return AgentIdentityFormat(
            id = agent_id,
            pos = _data['pos'],
            role = role,
            stats = CombatStats(hp = _data['hp'], max_hp=_data['max_hp'], skills=_data['skills'] )
        )
    
    @staticmethod
    def _get_data(role : str,grid_size) ->dict:
        if role == "Tank":
            return {
                "pos": (0,random.randint(2,grid_size-1)),
                "hp": 150,
                "max_hp" : 150,
                "skills" : {
                    "basic_attack" :Skill(power=10,min_range=1,max_range=1,cooldown=0),
                    "block" :Skill(power=0,min_range=0,max_range=0,cooldown=2),
                }
            }

        elif role == "Dealer":
            return {
                "pos": (random.randint(0,grid_size-1), 1),
                "hp": 80,
                "max_hp" : 80,
                "skills" : {
                    "basic_attack" :Skill(power=15,min_range=1,max_range=2,cooldown=0),
                    "special" :Skill(power=35,min_range=2,max_range=4,cooldown=3),
                }
            }
        
        elif role == "Healer":
            return {
                "pos": (random.randint(0,grid_size-1), 0),
                "hp": 60,
                "max_hp" : 60,
                "skills" : {
                    "heal" :Skill(power=-10,min_range=0,max_range=2,cooldown=1),
                    "all_heal" :Skill(power=-30,min_range=1,max_range=4,cooldown=7),
                }
            }
        elif role == "Boss":
            return {
                "pos": (random.randint(14,grid_size-1), random.randint(14,grid_size)),
                "hp": 300,
                "max_hp" : 300,
                "skills" : {
                    "basic_attack" :Skill(power=20,min_range=1,max_range=2,cooldown=1),
                    "aoe" :Skill(power=15,min_range=1,max_range=4,cooldown=5),
                }
            }

        return {}
