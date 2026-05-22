from .state import GameState 
from typing import List, Dict, Any
from engine.actions.action import ActionHandler
from engine.agents.base_agent import BaseAgent
from typing import Any, Dict, List
from reward import Reward_Calculator
import copy

class RaidEnv:
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.gamestate=None
        self.agents = {
            0: BaseAgent(0, "Tank", self.grid_size),
            1: BaseAgent(1, "Dealer", self.grid_size),
            2: BaseAgent(2, "Healer", self.grid_size),
            3: BaseAgent(3, "Boss", self.grid_size)
        }

        self.hero_roles = {"Tank":0, "Dealer":1, "Healer":2}
        self.boss_id = 3
        self.reward_calc = Reward_Calculator()
        self.step_count = 0

    def reset(self):
        self.gamestate = GameState(self.grid_size)

        for agent_id,agent in self.agents.items():
            team= "Boss" if agent.role=="Boss" else "Heroes"

            self.gamestate.register_agents(
                agent_id = agent_id,
                identity = agent.identity,
                team = team,
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

        step_rewards = {}   ## Simple dictionary for agents.. no need to initialize values.
        round_summary = {}   # Round summary
        agent_action_summary : Dict[int ,Dict[str, Any]] = {}    # Agent action summary - I was thinking to log the actions and see how everything is working out.. I need a agent action dict
        actions = {}

        alive_at_start = {aid: self.gamestate.is_alive(aid) for aid in self.agents.keys()}
        
        state_copy = copy.deepcopy(self.gamestate)

        for agent_id, agent in self.agents.items():  ## Having both agent id and agent object helps

            if not self.gamestate.is_alive(agent_id):  ## Iterate to the next agent if the current agent is not alive
                continue

            obs = self.get_obs_for_agents(agent_id)  ## Observation for a single agent.... 
            action_mask = self.gamestate.get_action_mask(agent_id)   #We are using a game state function to get the action mask .. which stores 1 for valid actions and 0 for invalid actions for that state.

            action = agent.get_action(obs, action_mask, is_training)  ##Store all actions inside a dictionary to send them to action sequencer to transition states directly at once
            actions[agent_id] = action  #Store them in dictionary with agent ids as keys

        step_summary = ActionSequencer.resolve_step(actions, self.gamestate)  ##Get the summaries from ActionSequencer

        #Need to calculate results next
        for agent_id , summary in step_summary.items():  ##Looping over all the step summary dictionary which contains all the things the agents did

            reward = self.reward_calc.calculate_reward(state_copy, self.gamestate, step_summary)
            step_rewards[agent_id] = reward   ##Store the reward for each agent..
            print(f'Step {self.step_count} ... {self.gamestate.identities[agent_id].role} is taking the action {actions[agent_id]} with reward {reward}')

            done = not self.gamestate.is_alive(agent_id)
            self.agents[agent_id].policy.store_reward(reward, done)

        env_done = self.gamestate.is_terminal()
        if env_done:
            for agent in self.agents.values():
                if len(agent.policy.memory["dones"]) > 0:
                    agent.policy.memory["dones"][-1] = True
        
        self.step_count += 1
        
        return self._get_all_observations(), step_rewards, env_done
                                                                                                                                                                                           
        
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
