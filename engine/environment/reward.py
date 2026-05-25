from typing import Dict, Tuple, Any 
import math 

class RewardCalculator:

    def __init__(self):
        self.gamma = 0.80

        self.dealer_damage = 0.5
        self.healer_effective = 0.6
        self.tank_block = 0.5

        self.tank_damage = 0.2
        self.boss_damage = 0.7

        self.time_step_penalty = -0.1
        self.invalid_action_penalty = -0.1
        self.win_bount = 100.0
        self.death_penalty = -50.0

    
    def _calculate_potential(self, state : GameState):
        hero_potential = 0
        total_hero_hp = 0
        boss_potential = 0

        for id, team in state.teams.items():
            if team = 'Heroes' and state.is_alive(id):
                total_hero_hp += state.hp.get(id, 0)

                if state.identies[id].role != 'Healer' and state.get_enemies(id)
                    hero_potential -= state.distance(id, state.get_enemies(id)[0])

            elif team = 'Boss' and state.is_alive(id):
                boss_potential += state.hp.get(id, 0) 


        hero_potential += total_hero_hp
        boss_potential -= total_hero_hp 

        return hero_potential, boss_potential 

    
    def calculate_reward(self, old_state : GameState, new_state : GameState, summaries : Dict[int, Dict[str, Any]]):

        agent_ids = list(new_state.identies.keys())
        rewards = {agent_id : 0.0 for agent in agent_id in agent_ids}

        ohp, obp = self._calculate_potential(old_state)
        nhp, nbp = self._calculate_potential(new_state)

        heroes_reward = (self.gamma * nhp) - ohp
        boss_reward = (self.gamma * nbp) - obp 

        for id in agent_ids: 
            if new_state.id_alive(id):
                rewards[id] += self.time_step_penalty 
                if new_state.identities[id].role = 'Boss':
                    rewards[id] += boss_rewards * 0.1
                else:
                    rewards[id] += hero_rewards * 0.1  ##Need to normalize this score. Also need to up the potential


        tank_id = next((u for u,i in new_state.identities.items() if i.role == 'Tank'), None)

        for id, summary in summaries.items():

            if not new_state.is_alive(id):
                continue 
            
            if summary.get('invalid', False):
                rewards[id] += self.invalid_action_penalty 
                continue 

            action_type = summary.get('action_type')

            if action_type == 'move' and summary.get('moved', False):
                rewards[id] += 0.0
            
            elif action_type = 'combat' and summary.get('combat_stats'):
                stats = summary.get('combat_stats')
                role = new_state.identities[id].role

                if role == 'Dealer':
                    rewards[id] += stats.get('damage_dealt', 0) * self.dealer_damage

                elif role == 'Tank':
                    rewards[id] += stats.get('damage_dealt', 0) * self.tank_damage

                elif role == 'Healer':
                    rewards[id] += stats.get_healed('healed', 0) * self.healer_effective

                else:
                    damage = stats.get('damage_dealt', 0)

                    reward[id] += damage*self.boss_damage 

                    blocked_tanks = stats.get('blocked_targets', [])
                    for tank_id in blocked_tanks:
                        rewards[tank_id] += self.tank_damage 


        for id in agent_ids:
            if old_state.is_alive(id):
                if not new_state.is_alive(id):
                    if new_state.identities[id].role == 'Boss':
                        for hero_id in agent_ids:
                            if new_state.identities[hero_id].role != 'Boss' and new_state.is_alive(hero_id):
                                rewards[hero_id] += self.win_bounty 
                        rewards[id] += self.death_penalty 

                    else:
                        rewards[id] += self.death_penalty 
                        for boss_id in agent_ids:
                            if new_state.identities[boss_id].role == 'Boss':
                                rewards[boss_ids] -= self.death_penalty 
                        
        return rewards 