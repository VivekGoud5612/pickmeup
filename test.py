import torch 
from engine.actions.action_sequencer import ActionSequencer 
from engine.agents.agent_data import AgentIdentityFormat, AgentRole, Teams, SkillTypes
from engine.actions.action import ActionTypes


ACTION_SKILL_MAP = {
    AgentRole.TANK : {
            ActionTypes.BASIC : SkillTypes.BASIC,
            ActionTypes.UTILITY : SkillTypes.BLOCK,
            ActionTypes.ULTIMATE : SkillTypes.INVINCIBLE,
        },

    AgentRole.DEALER : {
            ActionTypes.BASIC : SkillTypes.BASIC,
            ActionTypes.UTILITY : SkillTypes.PIERCE,
            ActionTypes.ULTIMATE : SkillTypes.SPECIAL,
        },

    AgentRole.HEALER : {
            ActionTypes.BASIC : SkillTypes.BASIC,
            ActionTypes.UTILITY : SkillTypes.HEAL,
            ActionTypes.ULTIMATE : SkillTypes.ALL_HEAL,
        },

    AgentRole.BOSS : {
            ActionTypes.BASIC : SkillTypes.BASIC,
            ActionTypes.UTILITY : SkillTypes.REGENERATE,
            ActionTypes.ULTIMATE : SkillTypes.AOE,
        },
    }

action_type = ActionType.BASIC 
role = AgentRole.TANK 

skill = ACTION_SKILL_MAP[role][action_type]

print(skill)