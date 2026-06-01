from typing import Dict,Tuple,Any
import math


class Reward_Calculator:
    def _init_(self,):
        self.gamma=0.99
        
        self.w_dealer_dmg = 0.5
        self.w_healer_eff = 0.6
        self.w_tank_block = 0.9

        self.w_tank_dmg=0.2
        self.w_boss_dmg = 0.7

        self.time_step_penalty = -0.1
        self.invalid_action_penalty = -0.1
        self.win_bounty = 100.0
        self.death_penalty = -50.0

    
    def calculate_potential(self,state):
        hero_potential=0
        total_hero_hp=0

        for u_id,team in state.teams.items():
            if team=="Heroes" and state.is_alive(u_id):
                total_hero_hp+=state.hp[u_id]
                
                if state.identities[u_id].role!="Healer" and state.get_enemies(u_id):
                    hero_potential-=state.distance(u_id,state.get_enemies(u_id)[0])
        
        hero_potential+=total_hero_hp

        boss_potential=0

        for u_id,team in state.teams.items():
            if team=="Boss" and state.is_alive(u_id):
                boss_potential+=state.hp[u_id]
            
        boss_potential-=total_hero_hp

        return hero_potential,boss_potential
    
    def calculate_reward(self, old_state, new_state, summaries : Dict[int, Dict[str, Any]]):
        
        agent_ids=list(new_state.identities.keys())
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
                rewards[id]+=0.00
            
            elif action_type=="combat" and summary.get("combat_stats"):
                stats=summary.get("combat_stats")
                role=new_state.identities[id].role

                if role=="Dealer":
                    rewards[id]+=stats.get("damage_dealt",0)*self.w_dealer_dmg
                elif role=="Healer":
                    rewards[id]+=stats.get("healed",0)*self.w_healer_eff
                elif role=="Tank":
                    if stats.get("damage_dealt",False):
                        rewards[id]+=stats.get("damage_dealt",0)*self.w_tank_dmg
                elif role=="Boss":
                    damage=stats.get("damage_dealt",0)

                    rewards[id]+=damage*self.w_boss_dmg
                            
                    blocked_tanks=stats.get("blocked_targets") or []
                    for t_id in blocked_tanks:
                        rewards[t_id]+=self.w_tank_block
                                                            
        
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









import torch






class Value_Normalizer:
    def __init__(self, epsilon : float = 1e-5):

        self.epsilon = epsilon

        self.running_mean = 0.0
        self.running_var = 1.0

        self.count = epsilon   #Global count for step rewards. Initialized to epsilon,to prevent zero division on first update


    def update(self, returns : torch.Tensor) -> None:

        batch_mean = torch.mean(returns).item()
        batch_var = torch.var(returns).item()
        batch_count = returns.numel()

        total_count = self.count + batch_count

        # Difference between this batch's mean and old historical mean
        delta = batch_mean - self.running_mean

        # Shift the running mean towards the new batch mean
        self.running_mean += delta * (batch_count/total_count)

        # Welford's and Chad's Algorithm
        # For running variance we calculate the sum of differences for both old data(ma) and new batch(mb).
        # Then we combine them using a correction factor for how much the mean just shifted (delta**2)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count)/total_count

        # Convert M2 to running variance 
        self.running_var = M2 / total_count

        self.count = total_count


    def Normalize(self, returns : torch.Tensor) -> torch.Tensor:
        # Standard deviation is square root of Variance
        std = (self.running_var + self.epsilon) ** 0.5

        return (returns - self.running_mean) / std
    

    def Denormalize(self, values : torch.Tensor) -> torch.Tensor:

        std = (self.running_var + self.epsilon) ** 0.5

        return (values * std) + self.running_mean