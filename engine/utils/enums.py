from enum import IntEnum 

class AgentID(IntEnum):
    TANK = 0
    DEALER = 1
    HEALER = 2
    BOSS = 3

class ActionTypes(IntEnum):  ## Now there is no need for action map or reverse action map...
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    WAIT = 4
    BASIC = 5
    UTILITY = 6
    ULTIMATE = 7

class AgentRole(IntEnum):
    TANK = 0
    DEALER = 1
    HEALER = 2
    BOSS = 3

class Teams(IntEnum):
    HEROES = 0
    MONSTERS = 1

class SkillTypes(IntEnum):
    BASIC = 0
    HEAL = 5
    BLOCK = 1
    AOE = 8
    SPECIAL = 4
    INVINCIBLE = 2
    REGENERATE = 7
    ALL_HEAL = 6
    PIERCE = 3

class AgentID(IntEnum):
    TANK = 0
    DEALER = 1
    HEALER = 2
    BOSS = 3

class ElementTypes(IntEnum):

    OBSERVATION = 0
    GLOBAL_STATE = 0
    ROLES = 0
    REWARDS = 0
    DONES = 0
    ACTION_MASKS = 0
    ACTIVE_MASKS = 0

class ActionTypes(IntEnum):  ## Now there is no need for action map or reverse action map...
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    WAIT = 4
    BASIC = 5
    UTILITY = 6
    ULTIMATE = 7