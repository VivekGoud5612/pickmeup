from typing import Dict, List, Any
class GameState:
    def __init__(self, agents : List[Dict]):
        self.agents = agents

    def get_agent(self, agent_id : int):
        return next((a for a in self.agents if a["id"] == agent_id), None)

    def get_enemies(self, agent_id : int):
        return [a for a in self.agents if a["id"] != agent_id]

    def is_terminal(self):
        alive = [a for a in self.agents if a["hp"] > 0]
        return len(alive) <= 1 #That is True if only one or no one is left and false if there are still agents left

    def distance(self, a_pos : tuple[int, int], b_pos : tuple[int, int]):
        return abs(a_pos[0] - b_pos[0]) + abs(a_pos[1] - b_pos[1])