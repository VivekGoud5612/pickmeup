from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True, kw_only=True)
class ReplayStartedResponse:
    """
    Response returned after a replay has been started.
    """

    message: str = "Replay started successfully."