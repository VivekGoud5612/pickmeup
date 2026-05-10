from typing import Dict, List, Any
from engine.agents.agent_data import AgentIdentity 


class GameState:

    def __init__(self):
        self.positions : Dict[int, tuple[int, int]] = {}
        self.hp : Dict[int, int] = {}
        self.teams : Dict[int, str] = {}
        self.identities : Dict[int, AgentIdentity] = {}


    def get_position(self, agent_id : int):
        return self.positions.get(agent_id)
    
    def get_hp(self, agent_id : int):
        return self.hp.get(agent_id)

    def get_identities(self, agent_id : int):
        return self.identities.get(agent_id)
    
    def is_alive(self, agent_id : int):
        return self.hp.get(agent_id) > 0

    def get_alive_agents(self):
        return [agent_id for agent_id, hp in self.hp.items() is hp > 0]

    def get_enemies(self, agent_id : int):
        agent_team = self.teams.get(agent_id)

        return (
            id
            for id , team in self.teams.items() if (team != agent_team and self.is_alive(id))
        )

    def is_terminal(self):
        alive_teams = {
            self.teams.get(agent_id) for agent_id in self.get_alive_agents()
        }
        return len(alive_teams) <= 1 #That is True if only one or no one is left and false if there are still agents left inside teams

    def distance(self, agent_ida : int, agent_idb :int):
        ax,ay = self.positions.get(agent_ida)
        bx,by = self.positions.get(agent_idb)
        return abs(ax - bx) + abs(ay - by)