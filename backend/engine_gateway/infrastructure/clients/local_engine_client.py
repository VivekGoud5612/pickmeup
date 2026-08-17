from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from backend.contracts.engine.dto.requests import (
    DeleteEngineCheckpointRequest,
    EvaluateEngineCheckpointRequest,
    InitializeEngineTrainingRequest,
    LoadEngineCheckpointRequest,
    PauseEngineTrainingRequest,
    ResumeEngineTrainingRequest,
    SaveEngineCheckpointRequest,
    StartEngineTrainingRequest,
    StopEngineTrainingRequest,
)
from backend.contracts.engine.dto.responses import (
    EngineCheckpointEvaluationResponse,
    EngineCheckpointSavedResponse,
    EngineMetricsResponse,
    EngineStatusResponse,
    EngineTrainingStartedResponse,
)
from backend.contracts.engine.enums import CheckpointType
from backend.engine_gateway.application.contracts.engine_client import EngineClient
from backend.engine_gateway.application.contracts.engine_event_publisher import (
    EngineEventPublisher,
)
from backend.engine_service.application.engine_service import LocalEngineService


class LocalEngineClient(EngineClient):
    """
    In-memory SDK adapter that talks to the local Engine Service facade.
    """

    def __init__(self, service: LocalEngineService, training_run_id) -> None:
        self._service = service
        self._training_run_id = training_run_id

    @property
    def _engine(self):
        return self._service.engine_for(self._training_run_id)

    def initialize(self, request: InitializeEngineTrainingRequest, publisher: EngineEventPublisher) -> None:
        if request.run_id != self._training_run_id:
            raise ValueError("Initialize request run_id does not match this client")
        self._service.initialize_training(request, publisher)

    def start(self, request: StartEngineTrainingRequest) -> EngineTrainingStartedResponse:
        worker = self._service.worker_for(self._training_run_id)
        worker.start()
        return EngineTrainingStartedResponse(status=worker.status)

    def pause(self, request: PauseEngineTrainingRequest) -> EngineStatusResponse:
        worker = self._service.worker_for(self._training_run_id)
        worker.pause()
        return EngineStatusResponse(status=worker.status)

    def resume(self, request: ResumeEngineTrainingRequest) -> EngineStatusResponse:
        worker = self._service.worker_for(self._training_run_id)
        worker.resume()
        return EngineStatusResponse(status=worker.status)

    def stop(self, request: StopEngineTrainingRequest) -> EngineStatusResponse:
        worker = self._service.worker_for(self._training_run_id)
        worker.stop()
        return EngineStatusResponse(status=worker.status)

    def save_checkpoint(self, request: SaveEngineCheckpointRequest) -> EngineCheckpointSavedResponse:
        checkpoint_name = request.checkpoint_name or "manual"
        checkpoint_path = Path(self._engine.checkpoint_directory) / checkpoint_name / f"step_{self._engine.global_step}.pt"
        self._engine.save_checkpoint(checkpoint_path, checkpoint_type=CheckpointType.MANUAL)
        return EngineCheckpointSavedResponse(
            checkpoint_path=checkpoint_path,
            checkpoint_type=CheckpointType.MANUAL,
            created_at=datetime.now(UTC),
        )

    def load_checkpoint(self, request: LoadEngineCheckpointRequest) -> None:
        self._engine.load_checkpoint(request.checkpoint_path)

    def delete_checkpoint(self, request: DeleteEngineCheckpointRequest) -> None:
        request.checkpoint_path.unlink(missing_ok=True)

    def evaluate_checkpoint(self, request: EvaluateEngineCheckpointRequest) -> EngineCheckpointEvaluationResponse:
        metrics = self._engine.evaluate(request.checkpoint_path)
        return EngineCheckpointEvaluationResponse(
            checkpoint_path=request.checkpoint_path,
            metrics=metrics,
        )

    def get_metrics(self, training_run_id) -> EngineMetricsResponse:
        if training_run_id != self._training_run_id:
            raise ValueError("Metrics requested for a different run")
        return EngineMetricsResponse(
            progress=self._engine.progress,
            metrics=self._engine.get_metrics(),
        )
