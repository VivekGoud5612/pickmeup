from engine.environment import GameState
from typing import Dict,List,Any

class ActionHandler:

    @staticmethod
    def _execute_movement(agent_id:int,action:int,gamestate:GameState)->bool:

        x,y=gamestate.positions[agent_id]

        if action==0:
            gamestate.positions[agent_id]=(x,y-1)
        elif action==1:
            gamestate.positions[agent_id]=(x,y+1)
        if action==2:
            gamestate.positions[agent_id]=(x-1,y)
        if action==3:
            gamestate.positions[agent_id]=(x+1,y)

        return True
    
    @staticmethod
    def _execute_combat(agent_id:int,action:int,gamestate:GameState)->Dict[str,Any]:
        
        result = {"damage_dealt": 0, "healed": 0, "blocked": False ,"target_id":None}

        identity=gamestate.identities[agent_id]
        role=identity.role

        skill_idx = action - 5
        skill_names = list(identity.stats.skills.keys())
        skill_name = skill_names[skill_idx]
        skill = identity.stats.skills[skill_name]

        gamestate.cooldowns[agent_id][skill_idx]=skill.cooldown

        gamestate.is_blocking[agent_id]=False

        enemies=gamestate.get_enemies[agent_id]

        if role in ["Tank","Dealer"]:
            if skill_name in ["basic_attack","special"] and enemies:
                boss_id=enemies[0]
                gamestate.hp[boss_id]-=skill.power
                result["damage_dealt"] = skill.power
            
            elif skill_name=="block":
                gamestate.is_blocking[agent_id] = True 
                result["blocked"] = True

        elif role=="Healer":
            my_team=gamestate.teams[agent_id]

            alive_allies=[
                uid for uid,team in gamestate.teams.items()
                if my_team==team and gamestate.is_alive(uid)
            ]
            
            if skill_name=="heal":
                lowest_hp_ally=None
                lowest_hp=float('inf')

                for a_id in alive_allies:
                    dist=gamestate.distance(agent_id,a_id)
                    if skill.min_range <= dist <=skill.max_range:
                        if gamestate.hp[a_id]<lowest_hp:
                            lowest_hp=gamestate.hp[a_id]
                            lowest_hp_ally=a_id
                
                if lowest_hp_ally is not None:
                    result["target_id"] = lowest_hp_ally
                    max_hp=gamestate.identities[lowest_hp_ally].stats.max_hp
                    gamestate.hp[lowest_hp_ally]=min(max_hp,lowest_hp-skill.power)
                    result["healed"]=abs(skill.power)
            
            elif skill_name=="all_heal":
                total_healed=0

                for a_id in alive_allies:
                    dist=gamestate.distance(agent_id,a_id)
                    if skill.min_range <= dist <= skill.max_range:
                        max_hp=gamestate.identities[a_id].stats.max_hp
                        gamestate.hp[a_id]=min(max_hp,gamestate.hp[a_id]-skill.power)
                        total_healed+=abs(skill.power)
                result["healed"]=total_healed

        elif role=="Boss":
            alive_heros=enemies

            if skill_name=="basic attack":
                closest_hero=None
                min_dist=float('inf')

                for h_id in alive_heros:
                    dist=gamestate.distance(agent_id,h_id)
                    if skill.min_range <= dist <=skill.max_range:
                        if dist<min_dist:
                            min_dist=dist
                            closest_hero =h_id

                if closest_hero is not None:
                    result["target_id"] = closest_hero
                    actual_damage=skill.power
                    if gamestate.identities[closest_hero].role=="Tank" and gamestate.is_blocking.get(closest_hero,False):
                        actual_damage=actual_damage/2
                    
                    gamestate.hp[closest_hero]-=actual_damage
                    result["damage_dealt"]=actual_damage

            elif skill_name=="aoe":
                total_damage=0

                for h_id in alive_heros:
                    dist=gamestate.distance(h_id,agent_id)
                    if skill.min_range<dist<skill.max_range:
                        actual_damage=skill.power
                        if gamestate.identities[h_id].role == "Tank" and gamestate.is_blocking.get(h_id, False):
                            actual_damage=actual_damage/2

                        gamestate.hp[h_id]-=actual_damage
                        total_damage+=actual_damage
                result["damage_dealt"]=total_damage
        
        return result


    @classmethod
    def perform_action(cls,agent_id:int,action:int,gamestate:GameState)->Dict[str,Any]:
        summary = {
            "action_type": None, 
            "moved": False, 
            "skipped": False, 
            "combat_stats": None
        }

        if action in [0, 1, 2, 3]:
            summary["action_type"] = "move"
            summary["moved"] = cls._execute_movement(agent_id, action, gamestate)
            
        elif action == 4:
            summary["action_type"] = "skip"
            summary["skipped"] = True
            
        elif action in [5, 6]:
            summary["action_type"] = "combat"
            summary["combat_stats"] = cls._execute_combat(agent_id, action, gamestate)

        return summary


