from __future__ import annotations

from enum import Enum, auto 

class CheckpointType(Enum):
    """
    Useful later, when we want to specify how we save checkpoints.
    It could be manual, periodic or best model save. These are 
    sort of choices or like flags to indicate that
    """
    MANUAL = auto()
    PERIODIC = auto()
    BEST_MODEL = auto() 
