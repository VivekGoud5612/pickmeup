import os
import time
import numpy as np
from collections import deque
from env import RaidEnv
from grid import Grid 

def train():
    print("Initializing Advanced Raid MARL Environment...")
    env = RaidEnv()
    vis = Grid(grid_size=10)
    
    # --- HYPERPARAMETERS ---
    MAX_EPISODES = 25000
    PPO_UPDATE_THRESHOLD = 2048   # Standard PPO batch size
    RENDER_INTERVAL = 1000        # Turn on Pygame every 1000 games
    LOG_INTERVAL = 100            # Print console stats every 100 games
    SAVE_INTERVAL = 5000          # Save weights every 5000 games
    MAX_ROUNDS = 100              # Prevent infinite games (kiting forever)
    
    # --- TRACKING METRICS ---
    global_steps = 0
    recent_rewards = {0: deque(maxlen=LOG_INTERVAL), 1: deque(maxlen=LOG_INTERVAL), 
                      2: deque(maxlen=LOG_INTERVAL), 3: deque(maxlen=LOG_INTERVAL)}
    recent_wins = deque(maxlen=LOG_INTERVAL) # 1 for Hero win, 0 for Boss win
    
    # Create directory for saving models
    os.makedirs("saved_models", exist_ok=True)

    print("\nStarting Training Loop...")
    for episode in range(1, MAX_EPISODES + 1):
        env.reset()
        done = False
        rounds_played = 0
        episode_rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        render_this_episode = (episode % RENDER_INTERVAL == 0)
        if render_this_episode:
            print(f"\n>>> RENDERING EVALUATION EPISODE {episode} <<<")

        # --- THE MATCH LOOP ---
        while not done and rounds_played < MAX_ROUNDS:
            # 1. Step the environment
            next_obs, rewards, done, summary = env.step(is_training=True)
            
            # 2. Accumulate metrics
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
            
            rounds_played += 1
            global_steps += 1
            
            # 3. Pygame Rendering (Slowed down slightly for human viewing)
            if render_this_episode:
                vis.render(env.gamestate)
                time.sleep(0.1) 

        # --- END OF MATCH PROCESSING ---
        # Determine who won
        boss_alive = env.gamestate.is_alive(env.boss_id)
        if not boss_alive:
            recent_wins.append(1) # Heroes win
        elif rounds_played >= MAX_ROUNDS:
            recent_wins.append(0) # Time out counts as Boss win/Draw
        else:
            recent_wins.append(0) # Boss wins
            
        # Log episode rewards
        for agent_id, reward in episode_rewards.items():
            recent_rewards[agent_id].append(reward)

        # --- PPO NETWORK UPDATE ---
        # Only trigger backpropagation if we have enough experiences gathered
        if global_steps >= PPO_UPDATE_THRESHOLD:
            for agent_id, agent in env.agents.items():
                if len(agent.policy.memory["states"]) > 0:
                    agent.policy.learn()
                agent.policy.clear_memory()
            global_steps = 0 # Reset counter after training

        # --- CONSOLE LOGGING ---
        if episode % LOG_INTERVAL == 0:
            avg_r = {i: np.mean(recent_rewards[i]) for i in range(4)}
            win_rate = np.mean(recent_wins) * 100
            
            print(f"Ep {episode:5d} | Hero Win Rate: {win_rate:5.1f}% | "
                  f"Avg Rewards -> Tank: {avg_r[0]:6.1f} | Dlr: {avg_r[1]:6.1f} | "
                  f"Hlr: {avg_r[2]:6.1f} | Boss: {avg_r[3]:6.1f}")

        # --- SAVING MODELS ---
        if episode % SAVE_INTERVAL == 0:
            for agent_id, agent in env.agents.items():
                role = env.gamestate.identities[agent_id].role
                agent.policy.save(f"saved_models/{role}_ep{episode}.pth")
            print(f"[SYSTEM] Checkpoint saved at Episode {episode}")

if __name__ == "__main__":
    train()