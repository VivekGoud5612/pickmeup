from __future__ import annotations 
from dataclasses import dataclass, field
from datetime import datetime, UTC 

from uuid import UUID, uuid4



@dataclass(slots = True, frozen = True, kw_only = True)
class EngineEvent:
    """
    A situation occurred with engine and we need to 
    let other services this, in this format
    """

    event_id : UUID = field(default_factory = uuid4())
    
    occured_at : datetime = field(default_factory = lambda : datetime.now(UTC))

 