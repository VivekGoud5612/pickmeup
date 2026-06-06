from dataclasses import dataclass
import random
from enum import IntEnum 

class AgentRole(IntEnum):
    TANK = 0
    DEALER = 1
    HEALER = 2
    BOSS = 3

class Teams(IntEnum):
    HEROES = 0
    VILLIANS = 1

@dataclass
class Skill:
    min_range:int
    max_range:int
    cooldown:int
    stamina_cost : int
    strength_of_skill : float

class SkillTypes(IntEnum):
    BASIC = 0
    HEAL = 5
    BLOCK = 1
    AOE = 8
    SPECIAL = 4
    AGGRO = 2
    REGENERATE = 7
    ALL_HEAL = 6
    PIERCE = 3

@dataclass 
class Attributes:
    strength : int 
    defence : int 
    stamina : int 
    recovery_rate : int 

@dataclass
class Stats:
    max_hp : int
    attributes : Attributes
    skills : dict[str,Skill]

@dataclass
class AgentIdentityFormat:
    id : int
    role: AgentRole
    stats : Stats

class AgentIdentity:
    ROLES : Dict[AgentRole, Dict[str, Any]] = {
        AgentRole.TANK : {
            'max_hp' : 200,
            'attributes' : Attributes(strength = 20, defence = 30, stamina = 90, recovery_rate = 4), ## Gives the rate at which stamina gets recovered.. Neeed to think of correct attrbutes as well now..
            'skills' : {
                SkillTypes.BASIC : Skill(min_range = 1, max_range = 1, cooldown = 0, stamina_cost = 5, strength_of_skill = 1.25),
                SkillTypes.BLOCK : Skill(min_range = 0, max_range = 0, cooldown = 10, stamina_cost = 10, strength_of_skill = 1.5), # Comparable to basic attack of dealer.. but no attack power, only able to block. But I dont know if I should another attack skill like dash or something
                SkillTypes.INVINCIBLE : Skill(min_range = 1, max_range = 2, cooldown = 20, stamina_cost = 35, strength_of_skill = 7),  #Need to model this wihtout an errors
            },
        },#Strength of a skill is similar to the rank of that skill. That is how much impact does that give ... 0-9

        AgentRole.DEALER : {
            "max_hp" : 140,
            'attributes' : Attributes(strength = 35, defence = 20, stamina = 100, recovery_rate = 5),
            "skills": {
                SkillTypes.BASIC : Skill(min_range = 1, max_range = 2, cooldown = 0, stamina_cost = 5, strength_of_skill = 1.5),
                SkillTypes.PIERCE : Skill(min_range = 1, max_range = 3, cooldown = 15, stamina_cost = 15, strength_of_skill = 1.75),
                SkillTypes.SPECIAL : Skill(min_range = 2, max_range = 4, cooldown = 20, stamina_cost = 40, strength_of_skill = 7), ## A very straining attack
            },
        },

        AgentRole.HEALER : {
            'max_hp' : 80,
            'attributes' : Attributes(strength = 15, defence = 20, stamina = 70, recovery_rate = 5),
            'skills' : {
                SkillTypes.BASIC : Skill(min_range = 1, max_range = 5, cooldown = 0, stamina_cost = 3, strength_of_skill = 1),
                SkillTypes.HEAL : Skill(min_range = 0, max_range = 2, cooldown = 2, stamina_cost = 3, strength_of_skill = 2),
                SkillTypes.ALL_HEAL : Skill(min_range = 1, max_range = 5, cooldown = 25, stamina_cost = 50, strength_of_skill = 9),
            },
        },

        AgentRole.BOSS : {
            "max_hp" : 1000,
            'attributes' : Attributes(strength = 40, defence = 30, stamina = 100, recovery_rate = 7),
            "skills": {
                SkillTypes.BASIC : Skill(min_range = 1, max_range = 2, cooldown = 3, stamina_cost = 5, strength_of_skill = 0.25),
                SkillTypes.REGENERATE : Skill(min_range = 0, max_range = 1, cooldown = 45, stamina_cost = 40, strength_of_skill = 6), ## Convert large amount of stamina to small hp that is proportional to strength * recovery here
                SkillTypes.AOE : Skill(min_range = 1, max_range = 4, cooldown = 35, stamina_cost = 35, strength_of_skill = 4.5),
            },
        },
    }

    @classmethod
    def create_identity(cls, agent_id : int, role: AgentRole):  ## For now , later we use this to create diverse agent identities and base agent takes that and 
         ## the rule based or LLM agent with some sort of identity

        data = cls.ROLES.get(role, None)

        if not data: 
            raise ValueError('Need a valid role')

        return AgentIdentityFormat(
            id = agent_id,
            role = role,
            stats = Stats(
                max_hp = data['max_hp'],
                attributes = data['attributes'],
                skills = data['skills'],
            ),
        )

