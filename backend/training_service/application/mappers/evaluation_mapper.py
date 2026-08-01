from __future__ import annotations

from backend.training_service.application.dto.replay.responses import (
    ReplayStartedResponse,
)


class ReplayMapper:
    """
    Maps replay results into response DTOs.
    """

    @staticmethod
    def to_started() -> ReplayStartedResponse:

        return ReplayStartedResponse(
            message="Replay started successfully.",
        )