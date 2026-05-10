from engine.agents.policies.base_policy import BasePolicy 
from engine.actions.attack import can_attack
from engine.actions.movement import move_towards
from engine.environment import GameState


class TankPolicy(BasePolicy):
    
    #def __init__(self, agent_id : int):
        #super().__init__(agent_id)

    def decide_action(self agent_id : int, state : GameState):

        enemies = state.get_enemies(agent_id)

        if not enemies:
            return ("STAY", )

        target = min(enemies, key = lambda enemy: state.distance(agent_id, enemy.id))   

        if can_attack(agent_id, target.id):
            return ('ATTACK', target.id)
        
        move = move_towards(state, agent_id, target.id)
        return ('MOVE', move)