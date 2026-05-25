from engine.agents.agent_data import AgentIdentity
from engine.agents.ppo.agent import Agent
from typing import List
from engine.environment.observation import Observation
from engine.agents.agent_data import AgentRole, SkillTypes, Skill
from engine.actions.action import ActionTypes

class BaseAgent:

    ACTION_MAP = {
        0: ActionTypes.UP,
        1: ActionTypes.DOWN,
        2: ActionTypes.RIGHT,
        3: ActionTypes.LEFT,
        4: ActionTypes.WAIT,
        5: ActionTypes.BASIC,
        6: ActionTypes.UTILITY,
        7: ActionTypes.ULTIMATE,
    }

    ACTION_SKILL_MAP = {
        AgentRole.TANK : {
            ActionTypes.BASIC : SkillTypes.BASIC,
            ActionTypes.UTILITY : SkillTypes.BLOCK,
            ActionTypes.ULTIMATE : SkillTypes.AGGRO,
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

    REVERSE_ACTION_MAP = {
        v : k for k,v in ACTION_MAP.items()
    }

    def __init__(self, agent_id :int, role : AgentRole):
        self.identity = AgentIdentity.create_identity(agent_id, role)

        self.id = agent_id
        self.role = role
        self.stats = self.identity.stats

    def get_action(self, observation : Observation, action_mask : List[int], is_training : bool):
        
        if self.policy is not None:
            action_idx = self.policy.get_action(observation, action_mask, is_training)
            action_type = self.ACTION_MAP[action_idx]
        else:
            action_type = ActionTypes.WAIT

        return action_type

    def spawn_configuration(self):
        self.current_hp = self.stats.max_hp 
        self.cooldowns = {
            some : 0 for some in self.stats.skills.keys()
        }
        self.stamina = self.stats.attributes.stamina 

    def is_alive(self):
        return self.current_hp > 0
        
    def take_damage(self, attack_power : float):

        self_defence = self.stats.attributes.defence 
        mitigation_multiplier = 100/(100 + self_defence)  ## USed in Games.. found it useful, and also attack power would be something like strength * strenght of skill or something which will be calculated in here..

        actual_damage = attack_power * mitigation_multiplier 
        self.current_hp = max(0, self.current_hp - actual_damage)

    def consume_stamina(self, action : ActionType):
        
        skill = self.ACTION_SKILL_MAP[self.role][action]
        stamina_cost = self.stats.skills[skill].stamina_cost
         
        if self.stamina < stamina_cost:
            raise RuntimeError("Stamina needs to be checked in Action Handler, this error shouldn't occur")

        self.stamina -= stamina_cost  ## Maybe we check for available stamina there in action handler. Or should we do it here?

    def update_cooldowns(self):
        
        for key in self.cooldowns.keys():
            self.cooldowns[key] = max(0, self.cooldowns[key] - 1) 
