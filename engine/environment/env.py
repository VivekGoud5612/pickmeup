from .grid import Grid 
from .state import GameState 
from typing import List, Dict, Any
from actions.action import ActionHandler
from agents.base_agent import BaseAgent

class Environment:
    def __init__(self,):
        self.grid = Grid(size = 10) 
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
        self.gamestate = GameState(self.grid)

        for agent_id,agent in self.agents.items():
            team="Boss" if agent.role=="Boss" else "Heroes"

            self.gamestate.register_agents(
                agent_id=agent_id,
                identity=agent.identity,
                team=team,
            )

            agent.policy.clear_memory()

        return self._get_all_observations
    

    def _get_all_observations(self):
        obs_dict = {}
        for uid, agent in self.agents.items():
            if agent.role == "Boss":
                obs_dict[uid] = self.gamestate.get_boss_observations(uid)
            else:
                obs_dict[uid] = self.gamestate.get_heroes_observations(uid, self.hero_roles, self.boss_id)
        return obs_dict

   
    def step(self,is_training:bool=True):

        total_rewards={0 : 0.0 ,1 : 0.0 ,2 :0.0 ,3 : 0.0}
        round_summary={}

        turn_order={0,1,2,3}

        for agent_id in turn_order:

            if not self.gamestate.is_alive(agent_id):
                continue

            agent=self.agents[agent_id]

            obs=self.get_obs_for_agents(agent_id)
            mask=self.gamestate.get_action_mask(agent_id)

            action=agent.get_action(obs,mask,is_training=is_training)

            summary=ActionHandler.perform_action(agent_id,action,self.gamestate)
            round_summary[agent_id]=summary

            reward=self.calculate_reward(agent_id,summary)
            total_rewards[agent_id]=reward

            done=self.gamestate.is_terminal()

            agent.policy.store_reward(reward,done)

            if done:
                break
        
        self.gamestate.update_cooldowns()

        return self._get_all_observations(), total_rewards, self.gamestate.is_terminal(), round_summary
    

    def get_obs_for_agents(self,agent_id):

        if self.agents[self.agent_id]=="Boss":
            return self.gamestate.get_boss_observations(agent_id)
        else:
           return self.gamestate.get_heroes_observations(agent_id,self.hero_roles,self.boss_id)
        
    
    def calculate_reward(self,agent_id:int,summary:Dict)->float:
        reward=-0.5
        role=self.agents[agent_id].role
        combat=summary.get("combat_stats")

        if combat:
            if role in ["Tank","Dealer"]:
                reward +=(combat["damage_dealt"]*0.5)
                if combat.get("blocked"):reward+=2.0

            elif role == "Healer":
                reward += (combat["healed"] * 0.7)
            
            elif role == "Boss":
                reward += (combat["damage_dealt"] * 0.8)