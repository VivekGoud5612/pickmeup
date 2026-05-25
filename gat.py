import torch 
import torch.nn as nn

class Actor_Critic(nn.Module):
    def __init__(self,state_size,action_size):
        super(Actor_Critic,self).__init__()

        self.actor=nn.Sequential(
            nn.Linear(state_size,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,action_size),
        )

        self.critic=nn.Sequential(
            nn.Linear(state_size,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,1),
        )

    def forward(self,state:torch.Tensor,mask:torch.Tensor):

        logits=self.actor(state)
        value=self.critic(state)

        if mask is not None:
            logits=logits.masked_fill(~mask,float('-inf'))

        return logits,value

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




        from dataclasses import dataclass,field
from enum import Enum
from typing import Dict,Any


class AgentRole(Enum):
    TANK = "Tank"
    DEALER = "Dealer"
    HEALER = "Healer"
    BOSS = "Boss"


@dataclass(frozen=True)
class Skill:
    power : int
    min_range : int
    max_range : int
    cooldown : int

@dataclass
class Stats:
    hp : int
    max_hp : int
    skills : Dict[str,Skill]

@dataclass
class AgentIdentityFormat:
    id : int
    role: AgentRole
    stats : Stats
    action_space_size : int


class AgentIdentity:

    ROLE_PRESETS : Dict[AgentRole , Dict[str , Any]] = {
        AgentRole.TANK : {
            "hp" : 150,
            "max_hp" : 150,
            "action_space_size" : 7,
            "skills" : {
                "basic_attack" : Skill(power = 10, min_range = 1, max_range = 1, cooldown = 0),
                "block" : Skill(power = 0, min_range = 0 ,max_range = 0, cooldown = 2),
            }
        },

        AgentRole.DEALER : {
            "hp" : 80,
            "max_hp" : 80,
            "action_space_size" : 7,
            "skills": {
                "basic_attack" : Skill(power = 15, min_range = 1, max_range = 2, cooldown = 0),
                "special" : Skill(power = 25, min_range = 2, max_range = 4, cooldown = 4),
            }
        },

        AgentRole.HEALER : {
            "hp" : 70,
            "max_hp" : 70,
            "action_space_size" : 7,
            "skills": {
                "basic_heal" : Skill(power = -10, min_range = 0, max_range = 2, cooldown = 2),
                "all_heal" : Skill(power = -30, min_range = 1, max_range = 4, cooldown = 10),
            }
        },

        AgentRole.BOSS : {
            "hp" : 1000,
            "max_hp" : 1000,
            "action_space_size" : 7,
            "skills": {
                "basic_attack" : Skill(power = 20, min_range = 1, max_range = 2, cooldown = 3),
                "aoe" : Skill(power = 40, min_range = 1, max_range = 4, cooldown = 15),
            }
        },
    }

    @staticmethod
    def create_identity(agent_id : int, role : AgentRole) -> AgentIdentityFormat:

        if role not in AgentIdentity.ROLE_PRESETS:
            raise ValueError(f"Role {role} is not registered in ROLE_PRESETS.")
        
        data = AgentIdentity.ROLE_PRESETS[role]

        stats = Stats(
            hp = data["hp"],
            max_hp = data["max_hp"],
            skills = data["skills"],
        )

        return AgentIdentityFormat(
            id = agent_id,
            role = role,
            stats = stats,
            action_space_size = data["action_space_size"]
        )
        