from engine.environment import Environment
from engine.agents import AgentIdentity, RuleAgent

def simulation():
    agI = AgentIdentity()
    agent_identity = agI.create_identity()
    env = Environment(agent_identity)
    state = env.reset()

    agents = [
        RuleAgent(1),
        RuleAgent(2),
    ]

    while True:

        actions = {}

        for agent in agents:
            action = actions[agent.id] = agent.act(state)
            state, done = env.step({agent.id : action})
            print(state.agents)

        if done:
            break
        
if __name__ == '__main__':
    simulation()