import torch 
import time 
import sys 
import torch 
from torch.utils.tensorboard import SummaryWriter 
import numpy as np 
import os
from collections import deque ## So that we can append fast.. We use this to track both episode rewardsand heroes win rate...
import traceback

from engine.environment.vector_env.shared_vector_env import SharedSubprocessVectorEnv
from engine.environment.env import Env 
from engine.agents.policy.trainer import MAgent
from engine.agents.policy.rollout import RolloutBuffer 
from engine.environment.observation import ObservationBuilder
from engine.environment.state import GameState

def make_env():
    return Env(grid_size = 10, max_steps = 200)  ## A simple function to instantiate the env...


def main():

    ##HYPERPARAMETERS 
    NUM_ENVS = 8
    NUM_STEPS = 200 
    NUM_AGENTS = 4
    TOTAL_TIMESTEPS = 5000000 ## For now 5000 steps.. let this run perfectly.. lets go to 5000000 - 5 mil.. model saves exactly 16 times..
    BATCH_SIZE = 1024  ## Let this be ...
    PPO_EPOCHS = 4
    INTENT_SIZE = 24
    CURRICULUM_LEVEL = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing MAPPO Training on {device}")

    ## Start the logger 
    run_name = f"MAPPO_GridWorld_{int(time.time())}" ## Standard format for any logger filess.. the file name is based on the time of thr running instance...
    writer = SummaryWriter(f"runs/{run_name}")

    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)  ## To save the model at 50 epochs (That is for 6400 * 5 steps)

    env_fns = [make_env for _ in range(NUM_ENVS)] ## I guess there is no need for () here..
    vector_env = SharedSubprocessVectorEnv(env_fns, num_agents = NUM_AGENTS)

    agent = MAgent(device = device)
    buffer = RolloutBuffer(
        num_steps = NUM_STEPS, num_envs = NUM_ENVS, num_agents = NUM_AGENTS,
        obs_shape = (ObservationBuilder.OBS_SIZE), intent_shape = (INTENT_SIZE), global_state_shape = (ObservationBuilder.GLOBAL_STATE_SIZE),
        actions_shape = (GameState.NUM_ACTIONS), device = device,
    )

    rolling_returns = deque(maxlen = 50)  ###  Keep a track of the last 50 games and then reset.. to store the next batch
    rolling_hero_win_rate = deque(maxlen = 50)
    rolling_boss_win_rate = deque(maxlen = 50)
    rolling_draw_rate = deque(maxlen = 50)
    rolling_boss_hp = deque(maxlen = 50)
    rolling_episode_length = deque(maxlen = 50)

    rolling_damage_dealt = deque(maxlen = 50)
    rolling_damage_blocked = deque(maxlen = 50)
    rolling_effectve_heal = deque(maxlen = 50)
    rolling_utility_uses = deque(maxlen = 50)
    rolling_ultimate_uses = deque(maxlen = 50)

    global_step = 0
    episode_number = 0
    start_time = time.time()  ## Start time

    try: ## To catch run time errors like in status error in shared env

        print("[*] Booting Shared Workers and Resetting Environments...")
        obs_dict, info = vector_env.reset()  ## The very first reset.. To just start with things.. Creates the first observation and such..

        current_obs = obs_dict['obs'] ## Current obs is just the pointer to liek store current obs and oter things
        current_global = obs_dict['global_state'] ## Shape (num_envs, num_agents, 96)
        current_action_masks = info['action_masks']
        current_active_masks = info['active_masks']

        print("[*] Training Loop Started.")  ## Start training loop.. we only reset once outside the loop, other resets are done in the step loop itself when all agents are dead or termina (one team dead) or truncated

        while global_step < TOTAL_TIMESTEPS:  # Till total time steps 

            ## PHASE 1 - Rollout... store
            for step in range(NUM_STEPS):
                global_step += (NUM_ENVS * NUM_AGENTS)  ## For all envs and for each agent ..so yeah


                actions, log_probs, values, intents = agent.get_actions_and_values(
                    obs = current_obs,
                    global_state = current_global,
                    action_masks =  current_action_masks,
                    is_training = True ## Sample randomly..          
                )

                ## STEP
                next_obs_dict, rewards, dones, truncated, next_info = vector_env.step(actions)

                ### Considering for last obs of a game and also our phase annealing thing...
                GAMMA = 0.95

                current_alpha = min(1.0 , global_step / TOTAL_TIMESTEPS) ### Alpha based phase annealing..

                old_pot = np.array([info.get('old_hand', np.zeros(NUM_AGENTS)) for info in next_info['env_infos']])  ##  A new way to unpack all the (8,4) elements (numenvs, num_agents)... We loop over num_envs to convert that dict to an array.. If there is no dict then we just a zeros array..
                new_pot = np.array([info.get('new_hand', np.zeros(NUM_AGENTS)) for info in next_info['env_infos']])

                handcrafted = (GAMMA * new_pot) - old_pot
                shaped_rewards = rewards + (1.0 - current_alpha) * handcrafted ## Now that alpha goes from 0 -1 smoothly as there are many steps...
                
                for env_idx in range(NUM_ENVS):
                    if truncated[env_idx]:  ## With the truncated information from above (which was an array converted in shared_env)
                        terminal_obs = next_info['env_infos'][env_idx]['terminal_observation'] ## Going to the very last terminal obs pf each env (num_agents, Dict[str, np.ndarray]), where that Dict would contain obs, state, role_ids

                        t_obs = np.expand_dims(terminal_obs['obs'], axis = 0) ## Expand the very first dim so the shape
                        t_global = np.expand_dims(terminal_obs['global_state_obs'], axis = 0)
                        t_action_masks = np.ones((1, NUM_AGENTS, GameState.NUM_ACTIONS), dtype = bool)  ## A simple toy mask to make just supplement for the actual mask as we only need the values here..

                        _, _, terminated_values, _ = agent.get_actions_and_values(
                            obs = t_obs, global_state = t_global,
                            action_masks = t_action_masks, is_training = False  ## No need for is training as we are calculating just the values..
                        )

                        shaped_rewards[env_idx] += GAMMA * terminated_values[0]  ## Just that as we are expanding the dims the shape of obs is (1, num_agents, 24) so the whole terminated_values come out to ne (1, num_agents) - so we take the first array of that ndarray so we get (num_agents)
                        dones[env_idx] = 1.0  ## We update that  although we have this in env.. nothing can go wrong to writing again..
                        ## No need for a new variable like next_dones, as these dones were calculated after 

                    env_info = next_info['env_infos'][env_idx]  

                 
                    if 'episode_length' in env_info:  ## if the info contains these we can say that the game is indeed over..
                        episode_number += 1
            
                        rolling_episode_length.append(env_info['episode_length'])

                        if 'boss_hp' in env_info:
                            rolling_boss_hp.append(env_info['boss_hp'])
                                
                        if 'episode_rewards' in env_info:
                                # Log the mean return of the agents for this episode
                            rolling_returns.append(np.mean(env_info['episode_rewards']))
                            rolling_damage_dealt.append(np.mean(env_info['damage_dealt']))
                            rolling_damage_blocked.append(np.mean(env_info['damage_blocked']))
                            rolling_effectve_heal.append(np.mean(env_info['effective_heal']))
                            rolling_utility_uses.append(np.mean(env_info['utility_uses']))
                            rolling_ultimate_uses.append(np.mean(env_info['ultimate_uses']))    
                        
                        if 'hero_win_rate' in env_info:
                            rolling_hero_win_rate.append(env_info['hero_win_rate'])
                        if 'boss_win_rate' in env_info:
                            rolling_boss_win_rate.append(env_info['boss_win_rate'])
                        if 'draw_rate' in env_info:
                            rolling_draw_rate.append(env_info['draw_rate'])

                        
                        if episode_number > 0 and episode_number % 10 == 0:
                            # Figure out the winner based on the flags
                            if env_info.get('hero_win_rate', 0.0) == 1.0:
                                winner = "Heroes"
                            elif env_info.get('boss_win_rate', 0.0) == 1.0:
                                winner = "Monsters"
                            else:
                                winner = "Draw"
                                    
                            ep_len = env_info['episode_length']
                            boss_hp_val = env_info.get('boss_hp', 0.0)
                                
                            # Extract arrays (Assuming indices: 0=Tank, 1=Dealer, 2=Healer, 3=Boss)
                            rews = env_info['episode_rewards']
                            dmg = env_info['damage_dealt']
                            blk = env_info['damage_blocked']
                            heal = env_info['effective_heal']
                            util = env_info['utility_uses']
                            ult = env_info['ultimate_uses']

                            print(f"\nEpisode {episode_number}")
                            print("-" * 42)
                            print(f"Length           : {ep_len}")
                            print(f"Winner           : {winner}")
                            print(f"Boss HP          : {boss_hp_val:.1f}")
                                
                            print("\nRewards")
                            print("-" * 8)
                            print(f"Tank             : {rews[0]:.1f}")
                            print(f"Dealer           : {rews[1]:.1f}")
                            print(f"Healer           : {rews[2]:.1f}")
                            print(f"Boss             : {rews[3]:.1f}")
                                
                            print("\nEpisode Statistics")
                            print("-" * 18)
                            print("Damage")
                            print(f"Tank             : {dmg[0]:.0f}")
                            print(f"Dealer           : {dmg[1]:.0f}")
                            print(f"Healer           : {dmg[2]:.0f}")
                            print(f"Boss             : {dmg[3]:.0f}")
                                
                            print("\nBlocked")
                            print(f"Tank             : {blk[0]:.0f}")
                            print(f"Dealer           : {blk[1]:.0f}")
                            print(f"Healer           : {blk[2]:.0f}")
                            print(f"Boss             : {blk[3]:.0f}")
                                
                            print("\nHealing")
                            print(f"Tank             : {heal[0]:.0f}")
                            print(f"Dealer           : {heal[1]:.0f}")
                            print(f"Healer           : {heal[2]:.0f}")
                            print(f"Boss             : {heal[3]:.0f}")
                                
                            print("\nUtility Uses")
                            print(f"Tank             : {util[0]:.0f}")
                            print(f"Dealer           : {util[1]:.0f}")
                            print(f"Healer           : {util[2]:.0f}")
                            print(f"Boss             : {util[3]:.0f}")
                                
                            print("\nUltimate Uses")
                            print(f"Tank             : {ult[0]:.0f}")
                            print(f"Dealer           : {ult[1]:.0f}")
                            print(f"Healer           : {ult[2]:.0f}")
                            print(f"Boss             : {ult[3]:.0f}\n")
                        
                            
                buffer.store(
                    local_obs = current_obs,
                    local_intents = intents,
                    global_state = current_global,
                    actions = actions,
                    log_probs = log_probs,  ## Log probs and others are from this current things.. but we store the next, obs, action mask and active mask so as to use it next time...
                    rewards = shaped_rewards,
                    dones = dones,
                    values = values,  ## Stores the current values.. The next value only becomes useful when computing shaped rewards or gae when truncated
                    action_masks = current_action_masks,
                    active_masks = current_active_masks,
                )  ## We just exclude returns advantages because they are calculated in computing gae
            
                current_obs = next_obs_dict['obs']  ## Only update the 
                current_global = next_obs_dict['global_state']
                current_action_masks = next_info['action_masks']
                current_active_masks = next_info['active_masks']  ## Just update the current running array and then use it in the next iteration
            
            ### PHASE 2 - GAE Calculation...
            _, _, next_values, _ = agent.get_actions_and_values(obs = current_obs, global_state = current_global, action_masks = current_action_masks, is_training = False) ## A way to just calculate the next value.. irrespective if the next step.. so that games where they ended naturally also get to know their next value, and not only the truncated ones.. The truncated ones are just used to add the total value to their rewards as they played well and they didnt die
            buffer.compute_returns_and_advantages(next_values = next_values, next_dones = dones)  ## Dones is the next dones, calculated after every step


            ### Phase 3 - PPO Update..
            avg_tank_actor_loss, avg_dealer_actor_loss, avg_healer_actor_loss, avg_boss_actor_loss, avg_entropy = 0.0, 0.0, 0.0, 0.0, 0.0 ## just initialize them..
            avg_tank_critic_loss, avg_dealer_critic_loss, avg_healer_critic_loss, avg_boss_critic_loss = 0.0, 0.0, 0.0, 0.0
            tank_advantages,  dealer_advantages, healer_advantages, boss_advantages = 0.0, 0.0, 0.0, 0.0
            update_steps = 0

            current_ent_coef = 0.05 * (1.0 - current_alpha)  ## We decay the entropy coefficient right here and send that as the parameter for update
            current_ent_coef = max(0.001, current_ent_coef)  ## IF that gets to 0, we allow a small exploration coeff
                
            data_generator = buffer.generate_batch(batch_size = BATCH_SIZE)  ## Note that this is a generator, so we need to loop over the generator to yield mini batches at a time..
            for mini_batch in data_generator: ## Note that the mini batch already contains the tensors so.. there is no need for another conversion
                loss_dict = agent.update(mini_batch, current_ent_coef)  ## loss dict automatically gets copied to CPU.. so there is no extra need for cpu memory transfer but the thing I dont understand is how does this generator work and is teh data generator something like an object, where we loop and the objects goes on yielding...

                avg_tank_actor_loss += loss_dict["actor_loss"][0] ## We have per agent losses
                avg_dealer_actor_loss += loss_dict['actor_loss'][1]
                avg_healer_actor_loss += loss_dict['actor_loss'][2]
                avg_boss_actor_loss += loss_dict['actor_loss'][3]

                avg_tank_critic_loss += loss_dict['critic_loss'][0]
                avg_dealer_critic_loss += loss_dict['critic_loss'][1]
                avg_healer_critic_loss += loss_dict['critic_loss'][2]
                avg_boss_critic_loss += loss_dict['critic_loss'][3]

                avg_entropy += loss_dict['entropy']
                update_steps += 1
                tank_advantages += mini_batch['advantages'][:, 0,].mean() ## We are tracking this as well, that is add each batchs advantage mean (That is add mean of 1024 advantages)... we add the means number of loop times (that is to the number of batches generated times)
                dealer_advantages += mini_batch['advantages'][:, 1,].mean()
                healer_advantages += mini_batch['advantages'][:, 2,].mean()
                boss_advantages += mini_batch['advantages'][:, 3,].mean()

            avg_tank_actor_loss /= update_steps ## We have per agent losses
            avg_dealer_actor_loss /= update_steps
            avg_healer_actor_loss /= update_steps
            avg_boss_actor_loss /= update_steps
            avg_tank_critic_loss /= update_steps
            avg_dealer_critic_loss /= update_steps
            avg_healer_critic_loss /= update_steps
            avg_boss_critic_loss /= update_steps
            avg_entropy /= update_steps

            tank_advantages /= update_steps ## We are tracking this as well, that is add each batchs advantage mean (That is add mean of 1024 advantages)... we add the means number of loop times (that is to the number of batches generated times)
            dealer_advantages /= update_steps
            healer_advantages /= update_steps
            boss_advantages /= update_steps
        
            buffer.clear() ### The training is done so we clear the buffer...


            ## Phase 4 Logging...
            sps = int(global_step / (time.time() - start_time))  ## I guess the steps per second or steps for this particular game to run... Not a single game but 8 different games with steps also counting for eacha agent..

            writer.add_scalar("Loss/TankActor", avg_tank_actor_loss, global_step)
            writer.add_scalar("Loss/DealerActor", avg_dealer_actor_loss, global_step)
            writer.add_scalar("Loss/HealerActor", avg_healer_actor_loss, global_step)
            writer.add_scalar("Loss/BossActor", avg_boss_actor_loss, global_step)

            writer.add_scalar("Loss/TankCritic", avg_tank_critic_loss, global_step)
            writer.add_scalar("Loss/DealerCritic", avg_dealer_critic_loss, global_step)
            writer.add_scalar("Loss/HealerCritic", avg_healer_critic_loss, global_step)
            writer.add_scalar("Loss/BossCritic", avg_boss_critic_loss, global_step)

            writer.add_scalar("Metrics/Entropy", avg_entropy, global_step) ## A graph to show the avg entropy over each time step in global steps
            writer.add_scalar("Metrics/SPS", sps, global_step)

            writer.add_scalar("Metrics/TankAdvantages", tank_advantages, global_step)
            writer.add_scalar("Metrics/DealerAdvantages", dealer_advantages, global_step)
            writer.add_scalar("Metrics/HealerAdvantages", healer_advantages, global_step)
            writer.add_scalar("Metrics/BossAdvantages", boss_advantages, global_step)


            ## Average reward will also be logged 
            writer.add_scalar("Environment/Mean_Reward", np.mean(buffer.rewards), global_step)  ## For all agents and for all the items in that batch

            writer.add_scalar("Environment/Episodic_Return", np.sum(rolling_returns), global_step)
            writer.add_scalar("Environment/Episode_Length", np.sum(rolling_episode_length), global_step)
            
            if len(rolling_hero_win_rate) > 0:
                writer.add_scalar("Team/Hero_Win_rate", np.mean(rolling_hero_win_rate), global_step)
                writer.add_scalar("Team/Boss_Win_rate", np.mean(rolling_boss_win_rate), global_step)
                writer.add_scalar("Team/Draw_rate", np.mean(rolling_draw_rate), global_step)

            if len(rolling_boss_hp) > 0:
                writer.add_scalar("Agent/Boss_HP", np.mean(rolling_boss_hp), global_step)

            if len(rolling_hero_win_rate) == 50: ## For every 50 games we do this
                current_win_rate = np.mean(rolling_hero_win_rate)

                if current_win_rate >= 0.85:  ## If the win rate for this specific 50 games exceed that threshold. then we can safely increase the difficulty
                    print(f"\n CURRICULUM LEVEL UP... Win rate hit {current_win_rate:.2f}")
                    CURRICULUM_LEVEL += 1

                    milestone_path = f"checkpoints/{run_name}/LEVEL_{CURRICULUM_LEVEL - 1}_Mastered.pt"  ## Save the model to this path for every win rate exceeding that win rate for this curriculum levels
                    torch.save({
                    # --- ACTORS ---
                    'tank_actor': agent.swarm.tank_actor.state_dict(),
                    'healer_actor': agent.swarm.healer_actor.state_dict(),
                    'dealer_actor': agent.swarm.dealer_actor.state_dict(),
                    'boss_actor': agent.swarm.boss_actor.state_dict(),
                    
                    # --- CRITICS ---
                    'tank_critic': agent.swarm.tank_critic.state_dict(),
                    'healer_critic': agent.swarm.healer_critic.state_dict(),
                    'dealer_critic': agent.swarm.dealer_critic.state_dict(),
                    'boss_critic': agent.swarm.boss_critic.state_dict(),
                    
                    # --- OPTIMIZERS (Crucial for resuming training later) ---
                    'tank_aoptim': agent.swarm.tank_aoptim.state_dict(),
                    'healer_aoptim': agent.swarm.healer_aoptim.state_dict(),
                    'dealer_aoptim': agent.swarm.dealer_aoptim.state_dict(),
                    'boss_aoptim': agent.swarm.boss_aoptim.state_dict(),
                    'tank_coptim': agent.swarm.tank_coptim.state_dict(),
                    'healer_coptim': agent.swarm.healer_coptim.state_dict(),
                    'dealer_coptim': agent.swarm.dealer_coptim.state_dict(),
                    'boss_coptim': agent.swarm.boss_coptim.state_dict(),
                    
                    }, milestone_path)

                    rolling_win_rate.clear()  ## Clear the rolling win rate so that we can stack that up again...

                    print(f"[*]Broadcasting Curriculum Level {CURRICULUM_LEVEL} to workers...")
                    current_obs_dict, current_info = vector_env.reset(curriculum_level = CURRICULUM_LEVEL)  ## We reset the env here with our new level and update those current obs andsuch with the new information

                    current_obs = current_obs_dict['obs']
                    current_global = current_obs_dict['global_state']
                    current_action_masks = current_info['action_masks']
                    current_active_masks = current_info['active_masks']

            if (global_step // (NUM_ENVS * NUM_AGENTS * NUM_STEPS)) % 10 == 0:  ## // divides and gives the nearest integer.. And also for each 10th step, I guess.. I dont know
                print(f"Step: {global_step} | SPS: {sps} | Ret: {np.mean(buffer.rewards):.2f} | Ent: {avg_entropy:.4f}")

            if (global_step // (NUM_ENVS * NUM_AGENTS * NUM_STEPS)) % 50 == 0:  ## Save the model for every 6400 * 50 steps.. at that path..
                save_path = f"checkpoints/{run_name}/step_{global_step}.pt"
                torch.save({
                    # --- ACTORS ---
                    'tank_actor': agent.swarm.tank_actor.state_dict(),
                    'healer_actor': agent.swarm.healer_actor.state_dict(),
                    'dealer_actor': agent.swarm.dealer_actor.state_dict(),
                    'boss_actor': agent.swarm.boss_actor.state_dict(),
                    
                    # --- CRITICS ---
                    'tank_critic': agent.swarm.tank_critic.state_dict(),
                    'healer_critic': agent.swarm.healer_critic.state_dict(),
                    'dealer_critic': agent.swarm.dealer_critic.state_dict(),
                    'boss_critic': agent.swarm.boss_critic.state_dict(),
                    
                    # --- OPTIMIZERS (Crucial for resuming training later) ---
                    'tank_aoptim': agent.swarm.tank_aoptim.state_dict(),
                    'healer_aoptim': agent.swarm.healer_aoptim.state_dict(),
                    'dealer_aoptim': agent.swarm.dealer_aoptim.state_dict(),
                    'boss_aoptim': agent.swarm.boss_aoptim.state_dict(),
                    'tank_coptim': agent.swarm.tank_coptim.state_dict(),
                    'healer_coptim': agent.swarm.healer_coptim.state_dict(),
                    'dealer_coptim': agent.swarm.dealer_coptim.state_dict(),
                    'boss_coptim': agent.swarm.boss_coptim.state_dict(),
                    
                }, save_path)
                print(f"[*] Brains & Optimizers saved to {save_path}")
                print(f"[*] Brain saved to {save_path}")

    # --- 5. CRITICAL CLEANUP ---
    except KeyboardInterrupt:
        print("\n[!] Training manually interrupted by user.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed:\n{e}")
        error_trace = traceback.format_exc()
        print(error_trace)

    finally:  ## Finally .. to be run regardless of the above exit code.. 
        print("[*] Cleaning up Shared Memory and closing workers...")
        vector_env.close()
        writer.close()
        print("[*] Shutdown complete. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()