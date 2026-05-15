from typing import Dict, List,Tuple, Any
from engine.agents.agent_data import AgentIdentity 
import numpy as np


class GameState:
    def __init__(self,grid_size:int=10):
        self.grid_size=grid_size

        self.positions :Dict[int ,Tuple[int,int]]={}
        self.hp :Dict[int,int]={}

        self.teams :Dict[int,str]={}
        self.identities :Dict[int,AgentIdentity]={}
        self.cooldowns :Dict[int,List[int]]={}
        self.is_blocking :Dict[int,bool]={}

    def register_agents(self,agent_id:int,identity:AgentIdentity,team:str):
        
        self.identities[agent_id]=identity
        self.positions[agent_id]=identity.pos
        self.hp[agent_id]=identity.stats.max_hp
        self.teams[agent_id]=team
        self.cooldowns[agent_id]=[0,0]
        self.is_blocking[agent_id]=False

    def is_alive(self,agent_id) ->bool:
        return self.hp.get(agent_id,0)>0

    def get_enemies(self,agent_id) ->List[int]:
        my_team=self.teams.get(agent_id)

        enemy_list=[uid for uid,team in self.teams.items()
                    if team!=my_team and self.is_alive(uid)]
        enemy_list.sort()

        return enemy_list
    
    def distance(self,id_a:int,id_b:int)->int:
        p1,p2=self.positions[id_a],self.positions[id_b]
        return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])
    
    def update_cooldowns(self):
        for agent_id in self.cooldowns:
            self.cooldowns[agent_id]=[max(0,cd-1) for cd in self.cooldowns[agent_id]]
    
    def is_terminal(self) ->bool:
        alive_team={self.teams[uid] for uid in self.hp if self.is_alive(uid)}

        return  len(alive_team)<=1
    

    def get_action_mask(self,agent_id:int) ->List[int]:

        mask=[1,1,1,1,1,1,1]
        ax,ay=self.positions[agent_id]

        if ay==0 :mask[0]=0
        if ay==self.grid_size-1 : mask[1]=0
        if ax==0 :mask[2]=0
        if ax==self.grid_size-1 :mask[3]=0

        enemies=self.get_enemies(agent_id)
        if not enemies:
            mask[5:]=[0,0]
            return mask

        skills=list(self.identities[agent_id].stats.skills.values())
        cds=self.cooldowns.get(agent_id,[0,0])

        for i in range(len(skills)):
            if cds[i]>0:
                mask[i+5]=0
                continue

            in_range=any(skills[i].min_range<=self.distance(agent_id,e_id)<=skills[i].max_range
                        for e_id in enemies)
            
            if not in_range:
                mask[i+5]=0

        return mask
    

    def get_heroes_observations(self,agent_id:int,hero_roles:Dict[str,int],boss_id:int) ->np.ndarray:

        obs=np.zeros(18,dtype=np.float32)
        my_role=self.identities[agent_id].role

        def fill_block(idx,uid):
            if uid is not None and self.is_alive(uid):
                ident = self.identities[uid]
                pos = self.positions[uid]
                cds = self.cooldowns.get(uid, [0, 0])
                obs[idx]   = self.hp[uid] / ident.stats.max_hp
                obs[idx+1] = pos[0] / self.grid_size
                obs[idx+2] = pos[1] / self.grid_size
                obs[idx+3] = 1.0 if cds[0] == 0 else 0.0
                obs[idx+4] = 1.0 if cds[1] == 0 else 0.0

        fill_block(0,agent_id)

        if my_role=="Tank": a,b=hero_roles.get("Dealer"),hero_roles.get("Healer")
        elif my_role=="Dealer": a,b=hero_roles.get("Tank"),hero_roles.get("Healer")
        else: a,b=hero_roles.get("Tank"),hero_roles.get("Dealer")

        fill_block(5,a)
        fill_block(10,b)

        if self.is_alive(boss_id):
            obs[15]=self.hp[boss_id]/self.identities[boss_id].stats.max_hp
            obs[16]=self.positions[boss_id][0] /self.grid_size
            obs[17]=self.positions[boss_id][1] /self.grid_size
        return obs
    

    def get_boss_observations(self,boss_id) ->np.ndarray:

        obs=np.zeros(18,dtype=np.float32)

        ident=self.identities[boss_id]
        pos=self.positions[boss_id]
        cds=self.cooldowns.get(boss_id,[0,0])

        obs[0]=self.hp[boss_id] / ident.stats.max_hp
        obs[1]=pos[0] /self.grid_size
        obs[2]=pos[1]  /self.grid_size
        obs[3]=1.0 if cds[0]==0 else 0.0
        obs[4]=1.0 if cds[1]==0 else 0.0

        heros=self.get_enemies(boss_id)

        for i in range(min(3, len(heros))):

            h_id=heros[i]
            start_idx=5 +(i*3)

            h_ident=self.identities[h_id]
            h_pos=self.positions[h_id]

            obs[start_idx]=self.hp[h_id] /h_ident.stats.max_hp
            obs[start_idx+1]=h_pos[0] /self.grid_size
            obs[start_idx+2]=h_pos[1] /self.grid_size

        return obs


        