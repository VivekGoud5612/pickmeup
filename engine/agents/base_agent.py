import os
import torch
from typing import Dict,Any
from engine.agents.agent_data import AgentIdentity,AgentRole
from engine.agents.ppo.agent import Agent


class BaseAgent:

    Fall_Back_Action=4

    def __init__(self, agent_id : int, role : AgentRole, obs_size : int = 18):

        self.identity = AgentIdentity.create_identity(agent_id, role)

        self.id = self.identity.id
        self.role = self.identity.role
        self.stats = self.identity.stats
        self.action_space_size = self.identity.action_space_size

        self.policy=Agent(obs_size, self.action_space_size, self.id, self.role)

        # Rolling runtime values unique to each agent instance
        self.current_hp: int = 0
        self.is_alive: bool = False
        self.is_blocking: bool = False
        self.cooldowns: Dict[str, int] = {}

        self.reset()


    def reset(self):

        self.current_hp = self.stats.max_hp
        self.is_alive = True
        self.is_blocking = False
        self.cooldowns = {skill_name : 0 for skill_name in self.stats.skills.keys()}

    def update_cooldowns(self) -> None:

        for skill_name in self.cooldowns:
            self.cooldowns[skill_name] = max(0, self.cooldowns[skill_name] - 1)

    def trigger_cooldowns(self, skill_name : str) -> None:

        if skill_name in self.cooldowns:
            self.cooldowns[skill_name] = self.stats.skills[skill_name].cooldown


    def take_damage(self, damage : int) -> int:

        if not self.is_alive:
            return 0
        
        final_damage = damage

        if self.is_blocking:
            final_damage = final_damage // 2
        
        self.current_hp = max(0, self.current_hp - final_damage)

        if self.current_hp == 0:
            self.is_alive = False

        return final_damage
    

    def take_heal(self, heal : int) -> int:

        if not self.is_alive:
            return 0
        
        before_hp=self.current_hp
        self.current_hp = min(self.stats.max_hp, self.current_hp - heal)

        return self.current_hp - before_hp
    

    def save_model(self, episode : int = 0, model_dir : str = "engine/saved_models") -> None:

        os.makedirs(model_dir, exist_ok = True)
        model_path = os.path.join(model_dir, f"{self.role.name}_ep{episode}.pth")

        self.policy.save(model_path)
        print(f"[SAVE] Saved weights for {self.role.name} at : {model_path}")

    
    def load_pretained_model(self, episode : int = 0, model_dir : str = "engine/saved_models") -> bool:

        model_path = os.path.join(model_dir, f"{self.role.name}_ep{episode}.pth")

        if os.path.exists(model_path):
            try:

                self.policy.policy.load_state_dict(torch.load(model_path, weights_only=True))
                print(f"[SUCCESS] Loaded Trained {self.role.name} from {model_path}")
                return True
            except Exception as e:

                raise RuntimeError(f"Failed to load corrupted model for {self.role.name}. Error :{e}")
            
        else:
            print(f"[INFO] No pretrained model for {self.role.name}. Starting fresh Training.")
            return False
                

    def get_action(self, observation : Any, action_mask : Any, is_training : bool = True) -> int:

        if self.policy is not None:
            return self.policy.get_action(observation, action_mask, is_training)

        else:
            print(f"[WARNING] Policy missing for {self.role.name}. Using fall back action.")
            return self.Fall_Back_Action
