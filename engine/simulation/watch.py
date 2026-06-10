import os
import time
import pygame
import torch
from engine.environment.env import RaidEnv
from engine.environment.grid import Grid

def watch():
    grid_size=9
    print("Initializing Advanced Raid MARL Watch Mode...")
    env = RaidEnv(grid_size)
    vis = Grid(grid_size)
    
    # --- SETTINGS ---
    CHECKPOINT_EPISODE = 25000
    MAX_ROUNDS_TO_WATCH = 4
    MAX_STEPS_PER_ROUND = 150 

    # Reset environment to initialize gamestate and agent identities properly
    print("Initializing environment states...")
    env.reset()

    # --- LOAD SAVED MODELS ---
    print(f"\nLoading weights from Episode {CHECKPOINT_EPISODE}...")
    for agent_id, agent in env.agents.items():
        role = env.gamestate.identities[agent_id].role
        model_path = f"saved_models/{role}_ep{CHECKPOINT_EPISODE}.pth"
        
        if os.path.exists(model_path):
            try:
                # Load weights into the actual torch network inside the policy wrapper
                agent.policy.policy.load_state_dict(torch.load(model_path))
                print(f"Loaded: {model_path}")
            except Exception as e:
                print(f"ERROR loading {model_path}: {e}")
                return
        else:
            print(f"ERROR: Could not find {model_path}. Check file names.")
            return
            
    print("\nAll brains loaded successfully. Starting evaluation...")
    time.sleep(1)

    # --- WATCH LOOP ---
    for round_num in range(1, MAX_ROUNDS_TO_WATCH + 1):
        env.reset()
        done = False
        steps = 0
        
        print(f"\n>>> STARTING ROUND {round_num} <<<")
        
        while not done and steps < MAX_STEPS_PER_ROUND:
            try:
                _, _, done = env.step(is_training=False) 
            except TypeError:
                _, _, done= env.step()
                
            steps += 1
            vis.render(env.gamestate)
            time.sleep(0.3) 

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exiting watch mode...")
                    pygame.quit()
                    return

        # Results logging
        boss_alive = env.gamestate.is_alive(env.boss_id)
        if not boss_alive:
            print(f"Round {round_num} Finished in {steps} steps -> HEROES WIN!")
        elif steps >= MAX_STEPS_PER_ROUND:
            print(f"Round {round_num} Finished in {steps} steps -> DRAW (Timeout)!")
        else:
            print(f"Round {round_num} Finished in {steps} steps -> BOSS WINS!")
            
        time.sleep(1.5) 

    print("\nEvaluation complete. Closing display.")
    pygame.quit()

if __name__ == "__main__":
    watch()