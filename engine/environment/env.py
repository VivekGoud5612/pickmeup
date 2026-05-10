from .grid import Grid 
from .state import GameState 
from typing import List, Dict, Any

class Environment:
    def __init__(self, agents :  List[Dict]):
        self.grid = Grid(size = 10) 
        self.agents = agents

    def reset(self):
        self.state = GameState(self.agents)
        return self.state

    def step(self, actions: Dict[int, str]):
        
        for agent in self.state.agents :
            if agent['hp'] <= 0:
                continue
            
            action = actions.get(agent['id'],None)

            if action is None:
                continue
            
            if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                new_agent_pos = self.grid.move(agent['pos'], action)
                agent['pos'] = new_agent_pos 

            elif action == 'ATTACK' :
                enemies = self.state.get_enemies(agent['id'])

                for enemy in enemies:
                    if self.state.distance(agent['pos'], enemy['pos']) <= 1:
                        enemy['hp'] -= agent['attack']
                    
            elif action == 'STAY':
                continue

        done = self.state.is_terminal()

        return self.state, done

    def state(self):
        return self.state