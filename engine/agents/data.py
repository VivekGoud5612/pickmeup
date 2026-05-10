from dataclasses import dataclass 

@dataclass
class CombatStats:
    hp: int
    attack: int
    min_range : int
    max_range : int