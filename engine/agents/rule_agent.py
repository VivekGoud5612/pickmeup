from .base_agent import BaseAgent 
from engine.environment.state import GameState

class RuleAgent(BaseAgent):
    def __init__(self, agent_id):
        super().__init__(agent_id)

    def act(self, state : GameState):
        agent = state.get_agent(self.id) 
        enemies = state.get_enemies(self.id)

        enemy = enemies[0]   #First enemy for now... but for multiple enemies check for the closes one

        if state.distance(agent['pos'], enemy['pos']) <= 1:
            return 'ATTACK'
        
        ax,ay = agent['pos']   ### If agent and enemy are not close then move the agent towards the enemy
        ex,ey = enemy['pos']  

        if ax != ex:
            return 'DOWN' if ax < ex else 'UP'
        elif ay != ey:
            return 'RIGHT' if ay < ey else 'LEFT'
        
        return 'STAY'