from __future__ import annotations 
from dataclasses import dataclass
from datetime import datetime, UTC 

from uuid import UUID, uuid4
from pydantic import field



@dataclass(slots = True, frozen = True, kw_only = True)
class EngineEvent:
    event_id : UUID = field(default_factory = uuid4())
    occured_at : datetime = field(default_factory = lambda : datetime.now(UTC))

 