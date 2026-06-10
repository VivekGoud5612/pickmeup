from dataclasses import dataclass,field
from enum import Enum
from typing import Dict,Any


class AgentRole(Enum):
    TANK = "Tank"
    DEALER = "Dealer"
    HEALER = "Healer"
    BOSS = "Boss"


@dataclass(frozen=True)
class Skill:
    power : int
    min_range : int
    max_range : int
    cooldown : int

@dataclass
class Stats:
    hp : int
    max_hp : int
    skills : Dict[str,Skill]

@dataclass
class AgentIdentityFormat:
    id : int
    role: AgentRole
    stats : Stats
    action_space_size : int


class AgentIdentity:

    ROLE_PRESETS : Dict[AgentRole , Dict[str , Any]] = {
        AgentRole.TANK : {
            "hp" : 150,
            "max_hp" : 150,
            "action_space_size" : 7,
            "skills" : {
                "basic_attack" : Skill(power = 10, min_range = 1, max_range = 1, cooldown = 0),
                "block" : Skill(power = 0, min_range = 0 ,max_range = 0, cooldown = 2),
            }
        },

        AgentRole.DEALER : {
            "hp" : 80,
            "max_hp" : 80,
            "action_space_size" : 7,
            "skills": {
                "basic_attack" : Skill(power = 15, min_range = 1, max_range = 2, cooldown = 0),
                "special" : Skill(power = 25, min_range = 2, max_range = 4, cooldown = 4),
            }
        },

        AgentRole.HEALER : {
            "hp" : 70,
            "max_hp" : 70,
            "action_space_size" : 7,
            "skills": {
                "basic_heal" : Skill(power = -10, min_range = 0, max_range = 2, cooldown = 2),
                "all_heal" : Skill(power = -30, min_range = 1, max_range = 4, cooldown = 10),
            }
        },

        AgentRole.BOSS : {
            "hp" : 1000,
            "max_hp" : 1000,
            "action_space_size" : 7,
            "skills": {
                "basic_attack" : Skill(power = 20, min_range = 1, max_range = 2, cooldown = 3),
                "aoe" : Skill(power = 40, min_range = 1, max_range = 4, cooldown = 15),
            }
        },
    }

    @staticmethod
    def create_identity(agent_id : int, role : AgentRole) -> AgentIdentityFormat:

        if role not in AgentIdentity.ROLE_PRESETS:
            raise ValueError(f"Role {role} is not registered in ROLE_PRESETS.")
        
        data = AgentIdentity.ROLE_PRESETS[role]

        stats = Stats(
            hp = data["hp"],
            max_hp = data["max_hp"],
            skills = data["skills"],
        )

        return AgentIdentityFormat(
            id = agent_id,
            role = role,
            stats = stats,
            action_space_size = data["action_space_size"]
        )