from enum import Enum, auto 

class TrainingStatus(Enum):
    """
    represents a lifecyle of training run.. 
    simply the status of each training run
    """

    CREATED = auto()
    RUNNING = auto()
    PAUSED = auto()
    FAILED = auto()
    COMPLETED = auto()


class TrainingAlgorithm(Enum):
    """
    Represent different RL algorithms which are supported
    """
    PPO = auto()
    MAPPO = auto()
    DQN = auto()
    SAC = auto()  ## Soft actor critic...


class EvaluationStatus(Enum):
    """
    State or Status of checkpoint evaluation
    """
    PENDING = auto() 
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()  ## But what does ti mean to evaluate a checkpoint ??  Suppose we after saving checkpoints, try to evaluate them to see which are better at which weights.. So these would act as choices for the conditions then


class CheckpointType(Enum):
    """
    Useful later, when we want to specify how we save checkpoints.
    It could be manual, periodic or best model save. These are 
    sort of choices or like flags to indicate that
    """
    MANUAL = auto()
    PERIODIC = auto()
    BEST_MODEL = auto() 
