from pathlib import Path

from backend.engine_gateway.application.dto.responses import (
    EngineCheckpointSavedResponse,
    EngineTrainingStartedResponse,
)

from backend.engine_gateway.application.dto.responses import (
    EngineStatusResponse,
    EngineMetricsResponse,
    EngineCheckpointEvaluationResponse,
)

from backend.training_service.domain.enums import (
    CheckpointType,
)

from engine.utils.enums import EngineStatus


class DummyEngineClient:

    def start(self, *args, **kwargs):

        return EngineTrainingStartedResponse(
            status=EngineStatus.RUNNING,
        )

    def pause(self, *args, **kwargs):

        return EngineStatusResponse(
            status=EngineStatus.PAUSED,
        )

    def resume(self, *args, **kwargs):

        return EngineStatusResponse(
            status=EngineStatus.RUNNING,
        )

    def stop(self, *args, **kwargs):

        return EngineStatusResponse(
            status=EngineStatus.STOPPED,
        )

    def save_checkpoint(
        self,
        checkpoint_name: str,
    ) -> EngineCheckpointSavedResponse:

        print(f"Saving checkpoint : {checkpoint_name}")

        return EngineCheckpointSavedResponse(
            checkpoint_path=Path(f"/tmp/{checkpoint_name}.pt"),
            checkpoint_type=CheckpointType.MANUAL,
        )

    def load_checkpoint(self, *args, **kwargs):
        pass

    def delete_checkpoint(self, *args, **kwargs):
        pass

    def evaluate_checkpoint(self, *args, **kwargs):

        return None

    def get_metrics(self, *args, **kwargs):

        return None