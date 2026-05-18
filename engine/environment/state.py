from typing import Dict, List, Tuple, Any
from engine.agents.agent_data import AgentIdentity 
import numpy as np

class GameState:
    def __init__(self, grid_size:int=10):
        self.grid_size = grid_size

        self.positions: Dict[int, Tuple[int,int]] = {}
        self.hp: Dict[int, int] = {}
        self.teams: Dict[int, str] = {}
        self.identities: Dict[int, AgentIdentity] = {}
        self.cooldowns: Dict[int, List[int]] = {}
        self.is_blocking: Dict[int, bool] = {}
        
        # --- NEW: Shared Team Map Tracking ---
        self.team_visited_tiles: Dict[str, set] = {"Heroes": set(), "Boss": set()}

    def register_agents(self, agent_id:int, identity:AgentIdentity, team:str):
        self.identities[agent_id] = identity
        self.positions[agent_id] = identity.pos
        self.hp[agent_id] = identity.stats.max_hp
        self.teams[agent_id] = team
        self.cooldowns[agent_id] = [0 for _ in identity.stats.skills]
        self.is_blocking[agent_id] = False
        
        # --- NEW: Record starting positions in the shared map ---
        self.team_visited_tiles[team].add(identity.pos)

    def is_alive(self, agent_id) -> bool:
        return self.hp.get(agent_id, 0) > 0

    def get_enemies(self, agent_id) -> List[int]:
        my_team = self.teams.get(agent_id)
        enemy_list = [uid for uid, team in self.teams.items()
                      if team != my_team and self.is_alive(uid)]
        enemy_list.sort()
        return enemy_list
    
    def distance(self, id_a:int, id_b:int) -> int:
        p1, p2 = self.positions[id_a], self.positions[id_b]
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def update_cooldowns(self):
        for agent_id in self.cooldowns:
            self.cooldowns[agent_id] = [max(0, cd - 1) for cd in self.cooldowns[agent_id]]
    
    def is_terminal(self) -> bool:
        alive_team = {self.teams[uid] for uid in self.hp if self.is_alive(uid)}
        return len(alive_team) <= 1
    
    def _hp_ratio(self, agent_id: int) -> float:
        ident = self.identities[agent_id]
        return max(0.0, min(1.0, self.hp[agent_id] / ident.stats.max_hp))

    def get_action_mask(self, agent_id:int) -> List[int]:
        mask = [1, 1, 1, 1, 1, 1, 1]
        ax, ay = self.positions[agent_id]

        occupied = [list(self.positions[uid]) for uid, hp in self.hp.items() if uid != agent_id and hp > 0]

        if ay == 0 or [ax, ay - 1] in occupied: mask[0] = 0
        if ay == self.grid_size - 1 or [ax, ay + 1] in occupied: mask[1] = 0
        if ax == 0 or [ax - 1, ay] in occupied: mask[2] = 0
        if ax == self.grid_size - 1 or [ax + 1, ay] in occupied: mask[3] = 0

        skills = list(self.identities[agent_id].stats.skills.items())
        cds = self.cooldowns.get(agent_id, [0 for _ in skills])
        my_team = self.teams[agent_id]
        enemies = self.get_enemies(agent_id)
        allies = [
            uid for uid, team in self.teams.items()
            if team == my_team and self.is_alive(uid)
        ]

        for i, (skill_name, skill) in enumerate(skills):
            if cds[i] > 0:
                mask[i + 5] = 0
                continue

            if skill_name in ["heal", "all_heal"]:
                in_range = any(
                    skill.min_range <= self.distance(agent_id, ally_id) <= skill.max_range
                    and self.hp[ally_id] < self.identities[ally_id].stats.max_hp
                    for ally_id in allies
                )
            elif skill_name == "block":
                in_range = True
            else:
                in_range = any(
                    skill.min_range <= self.distance(agent_id, e_id) <= skill.max_range
                    for e_id in enemies
                )
            
            if not in_range:
                mask[i + 5] = 0

        return mask

    def get_heroes_observations(self, agent_id:int, hero_roles:Dict[str,int], boss_id:int) -> np.ndarray:
        obs = np.zeros(18, dtype=np.float32)
        my_role = self.identities[agent_id].role

        def fill_block(idx, uid):
            if uid is not None and self.is_alive(uid):
                ident = self.identities[uid]
                pos = self.positions[uid]
                cds = self.cooldowns.get(uid, [0, 0])
                obs[idx]   = self._hp_ratio(uid)
                obs[idx+1] = pos[0] / self.grid_size
                obs[idx+2] = pos[1] / self.grid_size
                obs[idx+3] = 1.0 if cds[0] == 0 else 0.0
                obs[idx+4] = 1.0 if cds[1] == 0 else 0.0

        fill_block(0, agent_id)

        if my_role == "Tank": a, b = hero_roles.get("Dealer"), hero_roles.get("Healer")
        elif my_role == "Dealer": a, b = hero_roles.get("Tank"), hero_roles.get("Healer")
        else: a, b = hero_roles.get("Tank"), hero_roles.get("Dealer")

        fill_block(5, a)
        fill_block(10, b)

        # --- NEW: Shared Vision Radar Logic ---
        VISION_RANGE = 4
        boss_is_visible = False

        if self.is_alive(boss_id):
            # Scan all agents to find alive heroes and check their distance to the boss
            for uid, team in self.teams.items():
                if team == "Heroes" and self.is_alive(uid):
                    if self.distance(uid, boss_id) <= VISION_RANGE:
                        boss_is_visible = True
                        break  # Found him! No need to check other heroes

        if boss_is_visible:
            # Boss is spotted by the team! Share real statistics
            obs[15] = self._hp_ratio(boss_id)
            obs[16] = self.positions[boss_id][0] / self.grid_size
            obs[17] = self.positions[boss_id][1] / self.grid_size
        else:
            # Boss is hidden in the Fog of War
            obs[15] = 0.0   
            obs[16] = -1.0  
            obs[17] = -1.0  
            
        return obs
    
    def get_boss_observations(self, boss_id) -> np.ndarray:
        obs = np.zeros(18, dtype=np.float32)

        ident = self.identities[boss_id]
        pos = self.positions[boss_id]
        cds = self.cooldowns.get(boss_id, [0, 0])

        obs[0] = self._hp_ratio(boss_id)
        obs[1] = pos[0] / self.grid_size
        obs[2] = pos[1] / self.grid_size
        obs[3] = 1.0 if cds[0] == 0 else 0.0
        obs[4] = 1.0 if cds[1] == 0 else 0.0

        heros = self.get_enemies(boss_id)
        VISION_RANGE = 4

        for i ,h_id in enumerate([0,1,2]):
            start_idx = 5 + (i * 3)

            if self.is_alive(h_id) and self.distance(boss_id, h_id) <= VISION_RANGE:
                h_ident = self.identities[h_id]
                h_pos = self.positions[h_id]
                obs[start_idx]   = self._hp_ratio(h_id)
                obs[start_idx+1] = h_pos[0] / self.grid_size
                obs[start_idx+2] = h_pos[1] / self.grid_size
            else:
                # Hidden stats if out of range
                obs[start_idx]   = 0.0
                obs[start_idx+1] = -1.0
                obs[start_idx+2] = -1.0

        return obs
