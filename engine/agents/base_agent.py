from  engine.environment.state import GameState

class BaseAgent: ###base of all agents
    def __init__(self, agent_id : int):
        self.id = agent_id

    def act(self, state : GameState):
        raise NotImplementedError

if __name__ == '__main__':
    agent_id = 1
    agent = {}
    bd = BaseAgent(agent_id)
    state = GameState(agent)
    bd.act(state)