from engine.environment import GameState

def can_attack(state ; GameState, attacker_id : int, target_id : int):
    attacker = state.get_agent(attacker_id)

    distance = state.distance(attacker_id, target_id)

    stats = attacker.combat_stats 
    
    return ( stats.min_range <= distance <= stats.max_range )


def apply_attack(state : GameState, attacker_id : int, target_id : int):
    attacker = state.get_agent(attacker_id)
    target = state.get_agent(target_id)

    target.hp -= attacker.combat_stats.attack 
    return  