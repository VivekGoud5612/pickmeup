from __future__ import annotations

from backend.engine_gateway.application.dto.requests import (
    ReplayCheckpointRequest,
)

from engine.inference.inference import (
    Inference,
)


class ReplayEngineClient:
    """
    Lightweight client responsible for running replay inference.

    A new Inference engine is created for every replay request using
    the requested checkpoint.
    """

    def replay(
        self,
        request: ReplayEngineRequest,
    ) -> None:

        inference = Inference(
            checkpoint_path=request.checkpoint_path,
        )

        inference.replay(
            episodes=request.episodes,
        )