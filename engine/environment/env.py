from .state import GameState 
from typing import List, Dict, Any
from engine.actions.action import ActionHandler
from engine.agents.base_agent import BaseAgent

class RaidEnv:
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.gamestate=None
        self.agents = {
            0: BaseAgent(0, "Tank"),
            1: BaseAgent(1, "Dealer"),
            2: BaseAgent(2, "Healer"),
            3: BaseAgent(3, "Boss")
        }

        self.hero_roles={"Tank":0,"Dealer":1,"Healer":2}
        self.boss_id=3

    def reset(self):
        self.gamestate = GameState(self.grid_size)

        for agent_id,agent in self.agents.items():
            team="Boss" if agent.role=="Boss" else "Heroes"

            self.gamestate.register_agents(
                agent_id=agent_id,
                identity=agent.identity,
                team=team,
            )


        return self._get_all_observations()
    

    def _get_all_observations(self):
        obs_dict = {}
        for uid, agent in self.agents.items():
            if agent.role == "Boss":
                obs_dict[uid] = self.gamestate.get_boss_observations(uid)
            else:
                obs_dict[uid] = self.gamestate.get_heroes_observations(uid, self.hero_roles, self.boss_id)
        return obs_dict

   
    def step(self, is_training: bool = True):
        total_rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        round_summary = {}
        turn_order = [0, 1, 2, 3]
        
        # 1. Track exactly who was alive at the start of this round
        alive_at_start = {aid: self.gamestate.is_alive(aid) for aid in turn_order}

        # 2. Main Turn Loop for active agents
        for agent_id in turn_order:
            agent = self.agents[agent_id]
            
            # If the agent is dead before their turn, skip them completely
            if not self.gamestate.is_alive(agent_id):
                continue

            # Get current observations and action masks
            obs = self.get_obs_for_agents(agent_id)
            mask = self.gamestate.get_action_mask(agent_id)
            
            # Agent decides its action
            action = agent.get_action(obs, mask, is_training=is_training)

            # Execute the action inside the environment
            summary = ActionHandler.perform_action(agent_id, action, self.gamestate)
            round_summary[agent_id] = summary

            # Calculate base rewards (Boss kills are already natively calculated here)
            reward = self.calculate_reward(agent_id, summary)
            total_rewards[agent_id] = reward

            # Check if this specific action triggered match termination
            done = self.gamestate.is_terminal()
            
            # Store the standard step trajectory data
            agent.policy.store_reward(reward, done)

            # If an action ended the entire match, break the turn loop immediately
            if done:
                break

        # 3. --- ONE-TIME HERO DEATH PENALTY ---
        DEATH_PENALTY = -1.0  

        for agent_id in turn_order:
            agent = self.agents[agent_id]
            
            # Only apply if it's a Hero, they were alive at start, but are now dead
            if agent_id != self.boss_id and alive_at_start[agent_id] and not self.gamestate.is_alive(agent_id):
                # Apply penalty to environment step return dictionary
                total_rewards[agent_id] += DEATH_PENALTY 
                
                # Retroactively apply penalty to their last action's memory slot
                if len(agent.policy.memory["rewards"]) > 0:
                    agent.policy.memory["rewards"][-1] += DEATH_PENALTY
                    agent.policy.memory["dones"][-1] = True

        # 4. --- GLOBAL TERMINAL FALLBACK ---
        # If the match ended this round, find the surviving agents and close out their memory flags
        if self.gamestate.is_terminal():
            for agent_id in turn_order:
                agent = self.agents[agent_id]
                
                # If they survived the match but it abruptly ended, flip their last 'done' to True
                if self.gamestate.is_alive(agent_id):
                    if len(agent.policy.memory["dones"]) > 0:
                        agent.policy.memory["dones"][-1] = True

        # 5. Advance cooldowns and return normalized observations
        self.gamestate.update_cooldowns()
        return self._get_all_observations(), total_rewards, self.gamestate.is_terminal(), round_summary

    def get_obs_for_agents(self,agent_id):

        if self.agents[agent_id].role=="Boss":
            return self.gamestate.get_boss_observations(agent_id)
        else:
           return self.gamestate.get_heroes_observations(agent_id,self.hero_roles,self.boss_id)
        
    
    def calculate_reward(self,agent_id:int,summary:Dict)->float:
        reward=-0.01
        ident =self.gamestate.identities[agent_id]
        role=ident.role

        my_hp_ratio=self.gamestate.hp[agent_id]/ident.stats.max_hp
        boss_hp_ratio=self.gamestate.hp[self.boss_id]/self.gamestate.identities[self.boss_id].stats.max_hp

        if summary.get("action_type") == "move" and summary.get("moved") is True:
            reward += 0.02

        combat=summary.get("combat_stats") or{}
        target_id=combat.get("target_id")

        if role!="Boss" and combat.get("damage_dealt",0)>0:
            shared_team_reward=(combat["damage_dealt"] * 0.1)/10.0
            reward += shared_team_reward

        if combat:

            if role=="Dealer":
                dmg=combat.get("damage_dealt",0)

                multiplier=2.0 if boss_hp_ratio<0.3 else 1.0
                reward+=((dmg*multiplier*0.6))/10.0

            elif role=="Tank":
                if combat.get("blocked"):

                    reward+=(8.0 if boss_hp_ratio>0.5 else 4.0)/10.0

                reward+=((combat.get("damage_dealt",0)*0.3)/10.0)

            elif role=="Healer":
                heal_amt=combat.get("healed",0)

                if target_id is not None and heal_amt>0:
                    t_hp_ratio_before=combat.get("target_hp_ratio_before",1.0)

                    if t_hp_ratio_before<0.2:
                        reward+=2.0
                    else:
                        reward+=((heal_amt*0.8)/10.0)

                    if my_hp_ratio<0.25 and target_id!=agent_id:
                        reward-=0.5
            
            elif role=="Boss":
                dmg=combat.get("damage_dealt",0)

                if target_id is not None:
                    t_role=self.gamestate.identities[target_id].role
                    t_hp_ratio_before=combat.get("target_hp_ratio_before",1.0)

                    boss_reward=dmg*1.0

                    if t_role in ["Healer","Dealer"]:
                        boss_reward*=1.5

                    if t_hp_ratio_before<0.25:
                        boss_reward*=2.0
                    
                    reward+=(boss_reward/10.0)

                    if not self.gamestate.is_alive(target_id):
                        reward+=5.0
        
        return float(reward)
