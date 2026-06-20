import time
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Import your custom modules
from engine.environment.shared_vector_env import SharedSubprocessVectorEnv
from engine.environment.env import Env 
from engine.agents.policy.trainer import MAPPOAgent
from engine.agents.policy.rollout import RolloutBuffer

def make_env():
    return Env(grid_size=20, max_steps=200)

def main():
    # --- 1. Hyperparameters & Setup ---
    NUM_ENVS = 8
    NUM_STEPS = 200
    NUM_AGENTS = 4
    TOTAL_TIMESTEPS = 5_000_000
    BATCH_SIZE = 1024 # Or whatever fits your flat_size cleanly
    PPO_EPOCHS = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing MAPPO Training on {device}...")

    # Initialize Logger
    run_name = f"MAPPO_GridWorld_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Initialize Components
    env_fns = [make_env for _ in range(NUM_ENVS)]
    vector_env = SharedSubprocessVectorEnv(env_fns, num_agents=NUM_AGENTS)
    
    agent = MAPPOAgent(device=device)
    buffer = RolloutBuffer(
        num_steps=NUM_STEPS, num_envs=NUM_ENVS, num_agents=NUM_AGENTS,
        obs_shape=(24,), local_role_id_shape=(), global_state_shape=(96,), action_shape=(),
        device=device
    )

    global_step = 0
    start_time = time.time()

    try:
        # --- 2. Initial Reset ---
        print("[*] Booting Shared Workers and Resetting Environments...")
        obs_dict, info = vector_env.reset()
        
        current_obs = obs_dict['obs']
        current_global = obs_dict['global_state']
        current_roles = obs_dict['roles']
        current_action_masks = info['action_mask']
        current_active_masks = info['active_mask']
        
        print("[*] Training Loop Started.")
        
        # --- 3. Main Training Loop ---
        while global_step < TOTAL_TIMESTEPS:
            
            # --- PHASE 1: ROLLOUT ---
            for step in range(NUM_STEPS):
                global_step += (NUM_ENVS * NUM_AGENTS)
                
                # Get Actions (Fast Forward Pass)
                actions, log_probs, values = agent.get_actions_and_values(
                    obs=current_obs,
                    global_state=current_global,
                    roles=current_roles,
                    action_masks=current_action_masks,
                    is_training=True
                )
                
                # Step Environments
                next_obs_dict, rewards, dones, truncated, next_info = vector_env.step(actions)
                
                # Store Data
                buffer.store(
                    local_obs=current_obs,
                    local_ids=current_roles,
                    global_state=current_global,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    dones=dones,
                    values=values,
                    action_masks=current_action_masks,
                    active_masks=current_active_masks
                )
                
                # Update Pointers
                current_obs = next_obs_dict['obs']
                current_global = next_obs_dict['global_state']
                current_roles = next_obs_dict['roles']
                current_action_masks = next_info['action_mask']
                current_active_masks = next_info['active_mask']

            # --- PHASE 2: GAE CALCULATION ---
            # Bootstrap value for the last state
            _, _, next_values = agent.get_actions_and_values(
                obs=current_obs, global_state=current_global, 
                roles=current_roles, action_masks=current_action_masks, is_training=True
            )
            
            buffer.compute_returns_and_advantages(next_values=next_values, next_dones=dones)

            # --- PHASE 3: PPO UPDATE ---
            avg_actor_loss, avg_critic_loss, avg_entropy = 0.0, 0.0, 0.0
            update_steps = 0

            for _ in range(PPO_EPOCHS):
                data_generator = buffer.generate_batch(batch_size=BATCH_SIZE)
                for mini_batch in data_generator:
                    loss_dict = agent.update(mini_batch)
                    
                    avg_actor_loss += loss_dict['actor_loss']
                    avg_critic_loss += loss_dict['critic_loss']
                    avg_entropy += loss_dict['entropy']
                    update_steps += 1

            # Average out the losses for logging
            avg_actor_loss /= update_steps
            avg_critic_loss /= update_steps
            avg_entropy /= update_steps

            buffer.clear()

            # --- PHASE 4: LOGGING ---
            sps = int(global_step / (time.time() - start_time))
            
            writer.add_scalar("Loss/Actor", avg_actor_loss, global_step)
            writer.add_scalar("Loss/Critic", avg_critic_loss, global_step)
            writer.add_scalar("Metrics/Entropy", avg_entropy, global_step)
            writer.add_scalar("Metrics/SPS", sps, global_step)
            
            # Log average reward to see if they are actually learning
            writer.add_scalar("Environment/Mean_Reward", np.mean(buffer.rewards), global_step)

            if (global_step // (NUM_ENVS * NUM_AGENTS * NUM_STEPS)) % 10 == 0:
                print(f"Step: {global_step} | SPS: {sps} | Ret: {np.mean(buffer.rewards):.2f} | Act Loss: {avg_actor_loss:.4f} | Crit Loss: {avg_critic_loss:.4f} | Ent: {avg_entropy:.4f}")

    # --- 5. CRITICAL CLEANUP ---
    except KeyboardInterrupt:
        print("\n[!] Training manually interrupted by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed:\n{e}")
    finally:
        print("[*] Cleaning up Shared Memory and closing workers...")
        vector_env.close()
        writer.close()
        print("[*] Shutdown complete. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()