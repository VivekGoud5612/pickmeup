from dataclasses import dataclass


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
    role: str
    stats : CombatStats
    action_space_size:int=7

class AgentIdentity:
    def create_identity(self, agent_id : int, role: str = None, ):  ## For now , later we use this to create diverse agent identities and base agent takes that and 
         ## the rule based or LLM agent with some sort of identity
        
        _data = self._get_data(role,)
        return AgentIdentityFormat(
            id = agent_id,
            role = role,
            stats = CombatStats(hp = _data['hp'], max_hp=_data['max_hp'], skills=_data['skills'] )
        )
    
    @staticmethod
    def _get_data(role : str,) ->dict:
        if role == "Tank":
            return {
                "hp": 150,
                "max_hp" : 150,
                "skills" : {
                    "basic_attack" :Skill(power=10,min_range=1,max_range=1,cooldown=0),
                    "block" :Skill(power=0,min_range=0,max_range=0,cooldown=2),
                }
            }

        elif role == "Dealer":
            return {
                "hp": 80,
                "max_hp" : 80,
                "skills" : {
                    "basic_attack" :Skill(power=15,min_range=1,max_range=2,cooldown=0),
                    "special" :Skill(power=25,min_range=2,max_range=4,cooldown=3),
                }
            }
        
        elif role == "Healer":
            return {
                "hp": 70,
                "max_hp" : 70,
                "skills" : {
                    "heal" :Skill(power=-10,min_range=0,max_range=2,cooldown=2),
                    "all_heal" :Skill(power=-30,min_range=1,max_range=4,cooldown=10),
                }
            }
        elif role == "Boss":
            return {
                "hp": 1000,
                "max_hp" : 1000,
                "skills" : {
                    "basic_attack" :Skill(power=20,min_range=1,max_range=2,cooldown=3),
                    "aoe" :Skill(power=40,min_range=1,max_range=4,cooldown=15),
                }
            }

        return {}
