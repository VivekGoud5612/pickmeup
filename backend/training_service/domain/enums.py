from enums import Enum, auto 

class TrainingStatus(Enum):
    """
    represents a lifecyle of training run.. 
    simply the status of each training run
    """

    CREATED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto() 
    FALED = auto()


class TrainingAlgorithm(Enum):
    """
    Represent different RL algorithms which are supported
    """
    PPO = auto()
    MAPPO = auto()
    DQN = auto()
    SAC = ()  ## Soft actor critic...


class EvaluationStatus(Enum):
    """
    State or Status of checkpoint evaluation
    """
    PENDING = auto() 
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()  ## But what does ti mean to evaluate a checkpoint ??  Suppose we after saving checkpoints, try to evaluate them to see which are better at which weights.. So these would act as choices for the conditions then



