from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import EngineEvent
from ..enums import CheckpointType


@dataclass(slots = True, frozen = True, kw_only = True)
class CheckpointCreatedEvent(EngineEvent):

    checkpoint_path : Path 

    checkpoint_type : CheckpointType 

    global_step : int
