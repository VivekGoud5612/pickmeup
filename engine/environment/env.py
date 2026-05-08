from .grid import Grid 
from .state import GameState 

class Env:
    def __init__(self):
        self.grid = Grid(size = 10) 

    def reset(self);
        self.state = GameState(agents = 2)
        return self.state

    def step(actions: Dict[int, str]):
        
        for agent in self.state.agents :
            if agnet['hp'] <= 0:
                continue
            
            action = actions.get(agent['id'],None)

            if action is None:
                continue
            
            if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']
                new_agent_pos = self.grid.move(agent['pos'], action)
                agent['pos'] = new_agent_pos 

            elif action == 'ATTACK' :
                enemies = self.state.get_enemies(agent['id'])

                for enemy in enemies:
                    if self.state.distance(agent['pos'], enemy['pos']) <= 1:
                        enemy['hp'] -= agent['attack']

        done = self.state.is_terminal()

        return self.state, done

    def state(self):
        return self.state