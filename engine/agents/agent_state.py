from dataclasses import dataclass, field
from typing import Dict, Any
from engine.agents.agent_data import AgentIdentity

@dataclass
class AgentState:
    ##Pure data layer.. only contains agents runtime data.. will be given to GameState
    current_hp : float
    stamina : float
    cooldowns : Dict[str, int] = field(default_factory = dict)   ##Normal field validation of cooldown
    is_blocking : bool = False 


    @classmethod  # Doesnt have an instance, is only for the class directly
    def from_identity(cls, identity : AgentIdentity) -> AgentState:  ## A function to initiate all the run time attributes at the start of each episode..

        return cls(
            current_hp = identity.stats.max_hp,
            stamina = identity.stats.attributes.stamina,
            cooldowns = {skill : 0 for skill in identity.stats.skills.keys()}
        )

    @property
    def is_alive(self):
        return self.current_hp > 0

    @property
    def consume_stamina(self, stamina_cost : float):
         
        if self.stamina < stamina_cost:
            return False

        self.stamina -= stamina_cost  ## Maybe we check for available stamina there in action handler. Or should we do it here?
        return True

    @property
    def update_cooldown(cls):
        
        for key in self.cooldowns.keys():
            self.cooldowns[key] = max(0, self.cooldowns[key] - 1) 
