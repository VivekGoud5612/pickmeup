from engine.environment.state import GameState
from engine.agents.base_agent import BaseAgent
from pickmeup.engine.actions.action_handler import ActionHandler
from engine.environment.grid import Grid
import time


def dealer_training():
    gamestate=GameState(7)
    vis=Grid(7)
    ep=0

    agents={
        0:BaseAgent(0,"Dealer"),
        1:BaseAgent(1,"Boss"),
        2:BaseAgent(2,"Tank"),
    }
    heroes={
        "Tank":2,
        "Dealer":0,
    }

    for i in range(4):
        reset(gamestate,agents)

        episode_rewards=0
        done=False
        rounds=0

        while not done and rounds<50:

            obs=gamestate.get_heroes_observations(0,heroes,1)
            mask=gamestate.get_action_mask(0)

            dealer_action=agents[0].get_action(obs,mask,)
            tank_action=4
            boss_action=get_dummy_boss_action(1,0,gamestate)

            vis.render(gamestate)
            time.sleep(0.1)

            gamestate.update_cooldowns()


            summaries={}
            summaries[0]=ActionHandler.perform_action(0,dealer_action,gamestate)
            summaries[1]=ActionHandler.perform_action(1,boss_action,gamestate)
            summaries[2]=ActionHandler.perform_action(2,tank_action,gamestate)

            dealer_step_reward=calculate_dealer_sandbox_rewards(gamestate,summaries)

            # print(f"{gamestate.identities[0].role} took an action {healer_action} with reward :{healer_step_reward}")
            # print(f"{gamestate.identities[1].role} took an action {boss_action}")
            # print(f"{gamestate.identities[2].role} took an action {tank_action}")
            if not gamestate.is_alive(0) or not gamestate.is_alive(1) or not gamestate.is_alive(2):
                done = True
            
            agents[0].policy.store_reward(dealer_step_reward,done)

            episode_rewards+=dealer_step_reward
            rounds+=1
        print(f"Episode : {ep}")
        ep+=1
        agents[0].policy.learn()
        
    agents[0].policy.save("engine/sandboxes/Dealer_trained.pth")


def reset(gamestate:GameState,agents:dict):
    gamestate.identities.clear()
    gamestate.teams.clear()
    gamestate.positions.clear()
    gamestate.hp.clear()
    gamestate.cooldowns.clear()
    gamestate.is_blocking.clear()

    for id,agent in agents.items():
        team="Boss" if agent.role=="Boss" else "Heroes"
        gamestate.register_agents(
            agent_id=id,
            identity=agent.identity,
            team=team,

        )

def calculate_dealer_sandbox_rewards(gamestate,summaries):
    reward=-0.1

    dealer_stats=summaries[0].get("combat_stats") or {}
    if dealer_stats.get("damage_dealt",0)>0:
         reward+=2.0

    boss_stats=summaries[1].get("combat_stats") or {}
    if boss_stats.get("damage_dealt", 0) > 0:
        if boss_stats.get("target_id") == 0:
            reward -= 3.0

    if not gamestate.is_alive(0):
        reward-=5.0
    
    return reward


def get_dummy_boss_action(boss_id:int,target_id:int,gamestate:GameState):

    bx,by=gamestate.positions[boss_id]
    tx,ty=gamestate.positions[target_id]

    dist=gamestate.distance(boss_id,target_id)

    boss_identity=gamestate.identities[boss_id]
    skill_names=list(boss_identity.stats.skills.keys())
    skill=boss_identity.stats.skills[skill_names[0]]

    mask=gamestate.get_action_mask(boss_id)

    if skill.min_range <= dist <=skill.max_range:
        if mask[5]==1:
            return 5
    
    elif dist>skill.max_range:

        if bx>tx and mask[2]==1:
            return 2
        elif bx<tx and mask[3]==1:
            return 3
        

        elif by>ty and mask[0]==1:
            return 0
        elif by<ty and mask[1]==1:
            return 1
        
    return 4

def closest_enemy(boss_id:int,heroes:dict,gamestate:GameState)->int:
    closest_id=None
    min_dist=float('inf')

    for name,id in heroes.items():
        if gamestate.is_alive(id):
            dist=gamestate.distance(boss_id,id)
            
            if dist<min_dist:
                min_dist=dist
                closest_id=id
    
    return closest_id




if __name__=="__main__":
    dealer_training()