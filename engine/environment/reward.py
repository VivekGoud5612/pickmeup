from typing import Dict,Tuple,Any
import math


class Reward_Calculator:
    def __init__(self,):
        self.gamma=0.99
        
        self.w_dealer_dmg = 0.5
        self.w_healer_eff = 0.6
        self.w_tank_block = 0.3
        self.w_tank_dmg=0.2
        self.w_boss_dmg_tank = 0.3
        self.w_boss_dmg_squishy = 1.2
        self.w_tank_fail_protect = -1.0
        self.w_victim_damage = -0.5

        self.time_step_penalty = -0.01
        self.invalid_action_penalty = -0.1
        self.win_bounty = 100.0
        self.death_penalty = -50.0

    
    def calculate_potential(self,state):
        hero_potential=0
        total_hero_hp=0

        for u_id,team in state.teams.items():
            if team=="Heroes" and state.is_alive(u_id):
                total_hero_hp+=state.hp.get(u_id)
                hero_potential=total_hero_hp
                
                if state.identities[u_id].role!="Healer" and state.get_enemies(u_id):
                    hero_potential-=state.distance(u_id,state.get_enemies(u_id)[0])
            
            elif team=="Boss":
                boss_potential=state.hp[u_id]-total_hero_hp

        return hero_potential,boss_potential
    
    def calculate_reward(self,old_state,new_state,summaries:Dict[int,Dict[str,Any]])->Dict[int,float]:
        
        agent_ids=list(new_state.identites.keys())
        rewards={agent_id:0.0 for agent_id in agent_ids}


        #PBRS + Step Penalty:

        old_hero_phi , old_boss_phi =self.calculate_potential(old_state)
        new_hero_phi , new_boss_phi =self.calculate_potential(new_state)

        team_reward = (self.gamma * new_hero_phi) - old_hero_phi
        boss_reward = (self.gamma * new_boss_phi) - old_boss_phi

        for id in agent_ids:
            if new_state.is_alive(id):
                rewards[id] += self.time_step_penalty
                if new_state.identities[id].role=="Boss":
                    rewards[id]+=boss_reward*0.1
                else:
                    rewards[id]+=team_reward*0.1
        

        #Credit Assignment:

        tank_id = next((u for u, i in new_state.identities.items() if i.role == "Tank"), None)

        for id,summary in summaries.items():

            if not new_state.is_alive(id):
                continue

            if summary.get("invalid",False):
                rewards[id]+=self.invalid_action_penalty
                continue

            action_type=summary.get("action_type")

            if action_type=="move" and summary.get("moved",False):
                rewards[id]+=0.05
            
            elif action_type=="combat" and summary.get("combat_stats"):
                stats=summary.get("combat_stats")
                role=new_state.identities[id].role

                if role=="Dealer":
                    rewards[id]+=stats.get("damage_dealt",0)*self.w_dealer_dmg
                elif role=="Healer":
                    rewards[id]+=stats.get("healed",0)*self.w_healer_eff
                elif role=="Tank":
                    if stats.get("blocked",False):
                        rewards[id]+=self.w_tank_block
                    else:
                        rewards[id]+=stats.get("damage_dealt",0)*self.w_tank_dmg
                elif role=="Boss":
                    damage=stats.get("damage_dealt",0)
                    target_id=stats.get("target_id")

                    if damage>0 and target_id is not None:
                        rewards[target_id]+=damage*self.w_victim_damage
                        target_role=new_state.identities[target_id].role

                        if target_role=="Tank":
                            rewards[id]+=damage*self.w_boss_dmg_tank

                        elif target_role in ["Dealer","Healer"]:
                            rewards[id]+=damage*self.w_boss_dmg_squishy
                            if tank_id is not None and new_state.is_alive(tank_id):
                                rewards[tank_id]+=damage*self.w_tank_fail_protect
        
        #Sparse Rewards

        for id in agent_ids:
            if old_state.is_alive(id):
                if not new_state.is_alive(id):
                    if new_state.identities[id].role=="Boss":
                        for hero_id in agent_ids:
                            if new_state.identities[hero_id].role != "Boss" and new_state.is_alive(hero_id):
                                rewards[hero_id] += self.win_bounty
                        rewards[id]+=self.death_penalty

                    else:
                        rewards[id]+=self.death_penalty
                        for b_id in agent_ids:
                            if new_state.identities[b_id].role=="Boss":
                                rewards[b_id]-=self.death_penalty

        return rewards