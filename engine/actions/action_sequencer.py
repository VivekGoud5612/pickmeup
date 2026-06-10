from typing import Dict,Tuple,Any
from pickmeup.engine.actions.action_handler import ActionHandler

class Action_Sequencer:

    @staticmethod

    def resolve_step(action_dict:Dict[int,int],gamestate)->Dict[str,Dict[str,Any]]:

        summaries={}

#Phase-1
#Loop for to check whether tank used block,if so implement it
        for id,action in action_dict.items():
            role=gamestate.identities[id].role
            if role=="Tank" and action==6:
                summaries[id]=ActionHandler.perform_action(id,action,gamestate)

#Phase-2
#Loop for to check whether healer used heal,if so implement it
        for id,action in action_dict.items():
            role=gamestate.identities[id].role
            if id not in summaries and role=="Healer" and action==5:
                summaries[id]=ActionHandler.perform_action(id,action,gamestate)

#Phase-3 
#Loop for to check agents not in summaries if used movement,if so implement

        movement_intents={}
        
        for id,action in action_dict.items():
            if id not in summaries and action in [0,1,2,3,4]:
                target_pos=gamestate.get_target_position(id,action)

                if target_pos not in movement_intents:
                    movement_intents[target_pos]=[]
                    movement_intents[target_pos].append(id)
        
        for target_pos,ids in movement_intents.items():
                
                if len(ids)>1:
                    for id in ids:
                        summaries[id]={
                            "action_type": "move", 
                            "moved": False, 
                            "skipped": True, 
                            "invalid": False,
                            "combat_stats": None
                        }
                else:
                    for id in ids:
                        summaries[id]=ActionHandler.perform_action(id,action_dict[id],gamestate)

#Phase-6
#Loop for to check agents not in summary if used action==5,that is basic_attck,if so implement it
        for id,action in action_dict.items():
            role=gamestate.identities[id].role
            if id not in summaries and action==5:
                summaries[id]=ActionHandler.perform_action(id,action,gamestate)

#Phase-7     
#Loop for to check agents not in summary if used action==6,that is heavy attack,if so implement it
#Also check if alive ,since already implemented attck in previous phase
        for id,action in action_dict.items():
            role=gamestate.identities[id].role
            if id not in summaries and gamestate.is_alive(id) and action==6:
                summaries[id]=ActionHandler.perform_action(id,action,gamestate)

        return summaries