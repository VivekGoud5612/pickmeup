from __future__ import annotations

from datetime import UTC, datetime

from backend.engine_gateway.application.contracts.engine_client import (
    EngineClient,
)

from backend.engine_gateway.application.contracts.engine_event_publisher import (
    EngineEventPublisher,
)

from backend.engine_gateway.application.dto.requests import (
    InitializeEngineTrainingRequest,
    StartEngineTrainingRequest,
    PauseEngineTrainingRequest,
    ResumeEngineTrainingRequest,
    StopEngineTrainingRequest,
    SaveEngineCheckpointRequest,
    DeleteEngineCheckpointRequest,
    EvaluateEngineCheckpointRequest,
)

from backend.engine_gateway.application.dto.responses import (
    EngineTrainingStartedResponse,
    EngineStatusResponse,
    EngineMetricsResponse,
    EngineCheckpointSavedResponse,
    EngineCheckpointEvaluationResponse,
)

from backend.training_service.domain.enums import (
    CheckpointType,
)

from engine.training.training_engine import (
    TrainingEngine,
)


class LocalEngineClient(EngineClient):
    """
    Local implementation of EngineClient.

    One client communicates with exactly one TrainingEngine.
    """

    def __init__(
        self,
        engine: TrainingEngine,
    ) -> None:

        self._engine = engine


    #Initialize
    def initialize(
        self,
        request: InitializeEngineTrainingRequest,
        publisher: EngineEventPublisher,
    ) -> None:

        self._engine.initialize(
            config=request.configuration,
            run_id=request.run_id,
            checkpoint_directory=request.checkpoint_directory,
            checkpoint_path=request.checkpoint_path,
            publisher=publisher,
        )

    #Training
    def start(
        self,
        request: StartEngineTrainingRequest,
    ) -> EngineTrainingStartedResponse:

        self._engine.start()

        return EngineTrainingStartedResponse(
            status=self._engine.state,
        )

    def pause(
        self,
        request: PauseEngineTrainingRequest,
    ) -> EngineStatusResponse:

        self._engine.pause()

        return EngineStatusResponse(
            status=self._engine.state,
        )

    def resume(
        self,
        request: ResumeEngineTrainingRequest,
    ) -> EngineStatusResponse:

        self._engine.resume()

        return EngineStatusResponse(
            status=self._engine.state,
        )

    def stop(
        self,
        request: StopEngineTrainingRequest,
    ) -> EngineStatusResponse:

        self._engine.stop()

        return EngineStatusResponse(
            status=self._engine.state,
        )


    #Checkpoints
    def save_checkpoint(
        self,
        request: SaveEngineCheckpointRequest,
    ) -> EngineCheckpointSavedResponse:

        checkpoint_directory = (
            self._engine.checkpoint_directory /
            request.checkpoint_name
        )

        checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            checkpoint_directory /
            f"step_{self._engine.global_step}.pt"
        )

        self._engine.save_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_type=CheckpointType.MANUAL,
        )

        return EngineCheckpointSavedResponse(
            checkpoint_path=checkpoint_path,
            checkpoint_type=CheckpointType.MANUAL,
            global_step=self._engine.global_step,
            created_at=datetime.now(UTC),
        )


    def delete_checkpoint(
        self,
        request: DeleteEngineCheckpointRequest,
    ) -> None:

        checkpoint = request.checkpoint_path

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint}' does not exist."
            )

        if not checkpoint.is_file():
            raise ValueError(
                f"'{checkpoint}' is not a file."
            )

        checkpoint.unlink()

    def evaluate_checkpoint(
        self,
        request: EvaluateEngineCheckpointRequest,
    ) -> EngineCheckpointEvaluationResponse:

        metrics = self._engine.evaluate(
            request.checkpoint_path,
        )

        return EngineCheckpointEvaluationResponse(
            checkpoint_path=request.checkpoint_path,
            metrics=metrics,
        )

  
    #Metrics
    def get_metrics(
        self,
    ) -> EngineMetricsResponse:

        metrics = self._engine.get_metrics()

        return EngineMetricsResponse(
        metrics=metrics,
    )

    def replay(
        self,
        request: ReplayCheckpointRequest,
    ) -> None:
        """
        Runs inference on a checkpoint and streams frames
        to the frontend.
        """
        pass