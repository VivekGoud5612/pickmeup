from engine.agents.agent_data import AgentIdentity
from engine.agents.policy.agent import Agent
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
