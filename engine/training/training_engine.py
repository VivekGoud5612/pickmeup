from __future__ import annotations

import torch
from torch.uitls.tensorboard import SummaryWriter
from collections import deque 
import time
import sys 
import os 
import numpy as np 
import traceback 

from engine.environment.vector_env.shared_vector_env import SharedSubprocessVectorEnv
from engine.environment.state import GameState 
from engine.environment.env import Env 
from engine.agents.policy.trainer import MAgent 
from engine.agents.policy.rollout import RolloutBuffer 
from engine.environment.observation import ObservationBuilder 
from engine.utils.enums import EngineStatus 



class TrainingEngine:
    """
    A pure training class which contains all the attributes
    methdos required to initialize training, and do something to it..
    Note that this is object being used in the adapter between backend and 
    engine (local engine client)
    """

    def __init__(self) -> None:

        self._training_configuration = None 
        self._run_name = None 

        self._device = None

        self._magent = None
        self._buffer = None

        self._vector_env = None
        self._writer = None

        self._global_step = 0
        self._episode = 0

        self._state = EngineStatus.WAITING 

        self._initialized = False

    
    def initialize(
        self,
        config : TrainingConfiguration,   ## Contains all the hyper parameters, reward weights and curriculum settings.. .. 
        run_name : str,   ## from ENGINE request DTO.. will be used in locla engine client...
    ) -> None:

        self._training_configuration = config   ## create tensor board, env, agent, rollout, workers and such..
        self._run_name = run_name  ## Coming from backedn....

        self._initialize_runtime()   ## For run time stats, this keeps track of important things during run tiem, so that we can pause or resume
        self._initialize_logging()  
        self._initialize_checkpoint_directory()
        self._initialize_environment()
        self._initialize_agent()
        self._initialize_buffer()
        self._initialize_metrics()   ## Needed for update method... Both losses and advantages

        self._rollout_length = self._training_configuration.hyperparameters.rollout_length

        if self._state != EngineStatus.CREATED:
            raise RuntimeError("Engine has not been created")

        self._status = EngineStatus.INITALIZED

    def start(self):
        """
        Start training session. Here we can the training loop
        another fucntion which is in charge of rollout.
        Trainign continues till the stop or pause request is recieved
        """

        if self._state != EngineStatus.INITALIZED:
            raise RuntimeError("Training has not been initialized")

        self._state = EngineStatus.RUNNING 

        try :
            self._training_loop()

        finally:
            self.shutdown()  ## A public method can be used by other adapters or services...


    def pause(self):
        
        if self._state != EngineStatus.RUNNING:
            raise RuntimeError("Training needs to be running to pause")
        
        self._state = EngineStatus.PAUSED 

    def resume(self):
        
        if self._state != EngineStatus.PAUSED:
            raise RuntimeError("Training needs to be paused to resume")
        
        self._state = EngineStatus.RUNNING

    def stop(self):
        
        if self._state != EngineStatus.RUNNING:
            raise RuntimeError("Training needs to be running to stop")
        
        self._state = EngineStatus.STOPPED

    def save_checkpoint(self):
        ...
    
    def load_checkpoint(self):
        ...

    def evaluate(self):
        ...

    def get_metrics(self):
        ...

    def shutdown(self):
        ...

    def _training_loop(self) -> None:
        """
        Main training loop
        Cotnains rollout collection, calculating advantages, updating the network,
        and log metrics and save checkpoints as well.
        """
        self._reset_environment()    # A separate function to reset the environment

        while self._state != EngineStatus.STOPPED:   # Till we pause that... 

            self._collect_rollout()   ## Again neat and simple methods for each use...

            while self._state == EngineStatus.PAUSED:
                time.sleep(0.1)
            
            self._compute_advantages()

            self._update_policy()

            self._log_metrics()

            self._update_curriculum()

            self._periodic_checkpoint()

            if self._global_step >= self._training_configuration.hyperparameters.total_timesteps:
                self._state = EngineStatus.STOPPED   ## Could have wriiten break, but for the sake of while(condition)...


    def _reset_environment(self) -> None:

        obs, info = self._vector_env.reset()

        self._current_obs = obs['obs']
        self._current_state = obs['global_state']
        self._current_action_masks = info['action_masks']
        self._current_active_masks = info['active_masks']

    def _collect_rollout(self) -> None:
        """
        Collect one rollout and populate rollout buffer...
        Also check for terminal to store next values and update run time at last....
        """"

        for _ in range(self._rollout_length):   ## Game loop is here actually... global step loop is in training loop

            actions, values = self._magent.get_actions_and_values(
                obs = self._current_obs,
                global_state = self._current_state,
                action_masks = self._current_action_masks,
                is_training = self._training_configuration.is_training,
            )

            next_obs, self._dones, self._truncated, self._next_info = self._vector_env.step(actions)

            self.reward_annealing_alpha = min(1.0 , self._global_step / self._training_configuration.hyperparameters.total_timesteps) ### Alpha based phase annealing.. Need to model taht somehow...

            old_pot = np.array([info.get('old_hand', np.zeros(NUM_AGENTS)) for info in next_info['env_infos']])  ##  A new way to unpack all the (8,4) elements (numenvs, num_agents)... We loop over num_envs to convert that dict to an array.. If there is no dict then we just a zeros array..
            new_pot = np.array([info.get('new_hand', np.zeros(NUM_AGENTS)) for info in next_info['env_infos']])

            handcrafted = (self._training_configuration.hyperparameters.gamma * new_pot) - old_pot
            self._shaped_rewards = rewards + (1.0 - self.reward_annealing_alpha) * handcrafted ## Now that alpha goes from 0 -1 smoothly as there are many steps...
                
            self._bootstrap_truncated_episodes_and_push_logs(truncated, next_info)

            self._buffer.store(
                local_obs = self._current_obs,
                local_intents = intents,    ### For now these are non existent but there for future use
                global_state = current_global,
                actions = actions,
                log_probs = log_probs,  ## Log probs and others are from this current things.. but we store the next, obs, action mask and active mask so as to use it next time...
                rewards = shaped_rewards,
                dones = dones,
                values = values,  ## Stores the current values.. The next value only becomes useful when computing shaped rewards or gae when truncated
                action_masks = current_action_masks,
                active_masks = current_active_masks,
                )  ## We just exclude returns advantages because they are calculated in computing gae

            self._current_obs = next_obs_dict['obs']  
            self._current_global = next_obs_dict['global_state']
            self._current_action_masks = next_info['action_masks']
            self._current_active_masks = next_info['active_masks']


    def _bootstrap_truncated_episodes_and_push_logs(self):
        
        for env_idx in range(self._training_configuration.hyperparameters.num_envs):
            if self._truncated[env_idx]:  ## With the truncated information from above (which was an array converted in shared_env)
                terminal_obs = self._next_info['env_infos'][env_idx]['terminal_observation'] ## Going to the very last terminal obs pf each env (num_agents, Dict[str, np.ndarray]), where that Dict would contain obs, state, role_ids

                t_obs = np.expand_dims(terminal_obs['obs'], axis = 0) ## Expand the very first dim so the shape
                t_global = np.expand_dims(terminal_obs['global_state_obs'], axis = 0)
                t_action_masks = np.ones((1, NUM_AGENTS, GameState.NUM_ACTIONS), dtype = bool)  ## A simple toy mask to make just supplement for the actual mask as we only need the values here..

                _, _, terminated_values, _ = self._magent.get_actions_and_values(
                    obs = t_obs, global_state = t_global,
                    action_masks = t_action_masks, is_training = False  ## No need for is training as we are calculating just the values..
                )

                self._shaped_rewards[env_idx] += self._training_configuration.hyperparameters.gamma * terminated_values[0]  ## Just that as we are expanding the dims the shape of obs is (1, num_agents, 24) so the whole terminated_values come out to ne (1, num_agents) - so we take the first array of that ndarray so we get (num_agents)
                self._dones[env_idx] = 1.0  ## We update that  although we have this in env.. nothing can go wrong to writing again..
                ## No need for a new variable like next_dones, as these dones were calculated after 

            env_info = self._next_info['env_infos'][env_idx]  

            
            if 'episode_length' in env_info:  ## if the info contains these we can say that the game is indeed over..

                self._global_step += env_info['episode_length']  # Calculate global step in here, as we will get to know the actual step amount for each game of eahc env,....
                self._rolling_episode_length.append(env_info['episode_length'])

                if 'boss_hp' in env_info:
                    self._rolling_boss_hp.append(env_info['boss_hp'])
                        
                if 'episode_rewards' in env_info:
                        # Log the mean return of the agents for this episode
                    self._rolling_returns.append(np.mean(env_info['episode_rewards']))
                    self._rolling_damage_dealt.append(np.mean(env_info['damage_dealt']))
                    self._rolling_damage_blocked.append(np.mean(env_info['damage_blocked']))
                    self._rolling_effectve_heal.append(np.mean(env_info['effective_heal']))
                    self._rolling_utility_uses.append(np.mean(env_info['utility_uses']))
                    self._rolling_ultimate_uses.append(np.mean(env_info['ultimate_uses']))    
                
                if 'hero_win_rate' in env_info:
                    self._rolling_hero_win_rate.append(env_info['hero_win_rate'])
                if 'boss_win_rate' in env_info:
                    self._rolling_boss_win_rate.append(env_info['boss_win_rate'])
                if 'draw_rate' in env_info:
                    self._rolling_draw_rate.append(env_info['draw_rate'])

    def _compute_advantages(self):
        
        _, _, next_values, _ = self._magent.get_actions_and_values(obs = self._current_obs, global_state = self._current_state, action_masks = self._current_action_masks, is_training = False) ## A way to just calculate the next value.. irrespective if the next step.. so that games where they ended naturally also get to know their next value, and not only the truncated ones.. The truncated ones are just used to add the total value to their rewards as they played well and they didnt die
        self._buffer.compute_returns_and_advantages(next_values = next_values, next_dones = self.dones)  ## Dones is the next dones, calculated after every step

    def _update_policy(self):
        
        self._initialize_losses()

        current_ent_coef = 0.05 * (1.0 - self.reward_annealing_alpha)  ## Remove this form the training configs hyperparameters... this should be calculated from alpha ...
        current_ent_coef = max(0.001, current_ent_coef)  ## IF that gets to 0, we allow a small exploration coeff
        update_steps = 0

        data_generator = self._buffer.generate_batch(batch_size = self._training_configuration.hyperparameters.batch_size)
        for mini_batch in data_generator: ## Note that the mini batch already contains the tensors so.. there is no need for another conversion
            loss_dict = self._magent.update(mini_batch, current_ent_coef)  ## loss dict automatically gets copied to CPU.. so there is no extra need for cpu memory transfer but the thing I dont understand is how does this generator work and is teh data generator something like an object, where we loop and the objects goes on yielding...
            
            update_steps += 1
            self._calculate_losses(loss_dict)

        loss /= update_steps for loss in self._avg_losses[net] for net in ["actor", "critic"]
        self._avg_entropy /= update_steps

        self._advantages["tank"] /= update_steps ## We are tracking this as well, that is add each batchs advantage mean (That is add mean of 1024 advantages)... we add the means number of loop times (that is to the number of batches generated times)
        self._advantages["dealer"] /= update_steps
        self._advantages["healer"] /= update_steps
        self._advantages["boss"] /= update_steps


    def _calculate_losses(self, loss_dict):

        self._avg_losses["actor"]['tank'] += loss_dict["actor_loss"][0] ## We have per agent losses
        self._avg_losses["actor"]['dealer'] += loss_dict['actor_loss'][1]
        self._avg_losses["actor"]['healer'] += loss_dict['actor_loss'][2]
        self._avg_losses["actor"]['boss'] += loss_dict['actor_loss'][3]

        self._avg_losses["critic"]['tank'] += loss_dict['critic_loss'][0]
        self._avg_losses["critic"]['dealer'] += loss_dict['critic_loss'][1]
        self._avg_losses["critic"]['healer'] += loss_dict['critic_loss'][2]
        self._avg_losses["critic"]['boss'] += loss_dict['critic_loss'][3]

        self._avg_entropy += loss_dict['entropy']
        self._advantages['tank'] += mini_batch['advantages'][:, 0,].mean() ## We are tracking this as well, that is add each batchs advantage mean (That is add mean of 1024 advantages)... we add the means number of loop times (that is to the number of batches generated times)
        self._advantages['dealer'] += mini_batch['advantages'][:, 1,].mean()
        self._advantages['healer'] += mini_batch['advantages'][:, 2,].mean()
        self._advantages['boss'] += mini_batch['advantages'][:, 3,].mean()

    def _log_metrics(self):
        

    def _update_curriculum(self):
        ...

    def _periodic_checkpoint(self):
        ...

    def _initialize_runtime(self) -> None:
        
        self._global_step = 0
        self._episode = 0

        self._start_time = time.time()

        self._running = False
        self._paused = False
        
        self._current_obs = None
        self._current_state = None
        self._current_action_masks = None 
        self._current_active_masks = None

    def _initialize_logging(self) -> None:
        
        self._writer = SummaryWriter(
            f"runs/{self._run_name}"
        )

        self._rolling_returns = deque(maxlen=50)

        self._rolling_hero_win_rate = deque(maxlen=50)
        self._rolling_boss_win_rate = deque(maxlen=50)
        self._rolling_draw_rate = deque(maxlen=50)

        self._rolling_boss_hp = deque(maxlen=50)

        self._rolling_episode_length = deque(maxlen=50)

        self._rolling_damage_dealt = deque(maxlen=50)
        self._rolling_damage_blocked = deque(maxlen=50)
        self._rolling_effective_heal = deque(maxlen=50)

        self._rolling_utility_uses = deque(maxlen=50)
        self._rolling_ultimate_uses = deque(maxlen=50)

    
    def _initialize_checkpoint_directory(self) -> None:

        self._checkpoint_directory = (f"checkpoints/{self._run_name}")
        os.makedirs(self._checkpoint_directory, exist_ok = True)


    def _initialize_environment(self):

        env_fns = [self._create_environment for _ in range(self._training_configuration.hyperparameters.num_envs)]  ## from training config
        self._vector_env = SharedSubprocessVectorEnv(env_fns, num_agents = GameState.NUM_AGENTS)


    def _create_environment(self):

        return Env(
            grid_size = self._training_configuration.curriculum_settings.grid_size,
            max_steps = self._training_configuration.curriculum_settings.max_steps,
        )


    def _initialize_agent(self):
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._magent = MAgent(device = self._device)

    def _initialize_buffer(self):

        self._buffer = RolloutBuffer(
        num_steps = max_steps = self._training_configuration.hyperparameters.rollout_length,
        num_envs = self._training_configuration.hyperparameters.num_envs,
        num_agents = GameState.NUM_AGENTS,
        obs_shape = (ObservationBuilder.OBS_SIZE), intent_shape = (INTENT_SIZE), global_state_shape = (ObservationBuilder.GLOBAL_STATE_SIZE),
        actions_shape = (GameState.NUM_ACTIONS), device = self._device,
        )

        self._state = EngineStatus.CREATED 


    def _initialize_metrics(self) -> None:

        self._avg_losses : Dict[str, Dict[str, float]] = {
            "actor" : {
                role : 0.0 for role in ["tank", "dealer", "healer", "boss"]
            },
            "critic" : {
                role : 0.0 for role in AgentRole
            },
        }
        
        self._advantages : Dict[str, float] = {
            role : 0.0 for role in ["tank", "dealer", "healer", "boss"]
        }

        self._avg_entropy = 0


    def _cleanup(self):
        ...