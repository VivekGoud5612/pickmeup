from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(slots=True, frozen=True, kw_only=True)
class ReplayCheckpointRequest:
    """
    Replay a checkpoint by streaming inference frames
    to the frontend.
    """

    checkpoint_id: UUID

    replay_episodes: int = 1