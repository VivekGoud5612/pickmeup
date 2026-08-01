from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True, kw_only=True)
class EngineEvent:
    """
    Marker base class for all engine events.
    """
    pass