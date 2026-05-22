from engine.agents.agent_data import AgentIdentity
from engine.agents.ppo.agent import Agent
from typing import List
from engine.environment.observation import Observation
class BaseAgent:
    def __init__(self, agent_id :int, role :str, gamestate : GameState):
        
        identity_factory = AgentIdentity()
        self.identity = identity_factory.create_identity(agent_id, role)

        self.id = self.identity.id
        self.role = self.identity.role
        self.stats = self.identity.stats
        self.action_space_size = self.identity.action_space_size

        dummy_obs = ObservationEncoder.build_observation(self.id, gamestate)
        obs_size = len(dummy_obs.to_vector())
        self.policy = AgentPolicy(obs_size, self.identity.action_space_size)

    def get_action(self, observation : Observation, action_mask : List[int], is_training : bool):
        
        if self.policy is not None:
            action = self.policy.get_action(observation, action_mask, is_training)
        else:
            action = 4

        return action
