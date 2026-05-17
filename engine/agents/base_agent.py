from agent_data import AgentIdentity
from PPO.agent import Agent

class BaseAgent:
    def __init__(self,agent_id :int ,role :str):
        identity_factory=AgentIdentity()
        self.identity=identity_factory.create_identity(agent_id,role)

        self.id=self.identity.id
        self.role=self.identity.role
        self.pos=list(self.identity.pos)
        self.stats=self.identity.stats
        self.action_space_size=self.identity.action_space_size

        self.policy=Agent(18,self.action_space_size,self.id,self.role)


    def get_action(self,observation,action_mask,is_training):
        if self.policy is not None:
            action=self.policy.get_action(observation,action_mask,is_training)

        else:
            action=4
        return action
