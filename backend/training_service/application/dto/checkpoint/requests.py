from __future__ import annotations 
from dataclasses import dataclass 
from uuid import UUID



@dataclass(slots = True, frozen = True, kw_only = True)
class SaveCheckpointRequest:
    """
    Request to save a checkpoint
    """

    training_run_id : UUID 

    checkpoint_name : str

    # episode : int   ## Note that there are no weights as the backend already considers that, and we can get them from training_run_id.config_id.weights (hyper or rew or cur)

    # step : int 

    notes : str = ""

@dataclass(slots=True, frozen=True, kw_only=True)
class GetCheckpointRequest:

    checkpoint_id: UUID

@dataclass(slots=True, frozen=True, kw_only=True)
class GetLatestCheckpointRequest:

    training_run_id: UUID  ## Get best checkpoint of that training run

@dataclass(slots=True, frozen=True, kw_only=True)
class GetBestCheckpointRequest:

    training_run_id: UUID

@dataclass(slots=True, frozen=True, kw_only=True)
class DeleteCheckpointRequest:

    checkpoint_id: UUID

@dataclass(slots=True, frozen=True, kw_only=True)
class ListCheckpointsRequest:

    training_run_id: UUID

