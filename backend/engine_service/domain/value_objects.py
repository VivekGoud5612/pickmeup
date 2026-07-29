from __future__ import annotations

from dataclasses import dateclass, field 
from datetime import datetime, UTC 
from pathlib import Path 



@dataclass(slots = True, frozen = True, kw_only = True)
class CheckpointMetaData:
    """
    just some data regarding the checkpoint saved
    . Can be used afterwards if we wish to send to backend.
    Also to persist the time and size along with when with the checkpoint path
    """

    path : Path 

    created_at : datetime = field(default_factory = lambda : datetime.now(UTC))

    size_bytes : int 

    episodes : int