from __future__ import annotations 

from abc import ABC, abstractmethod 

from training_service.application.dto.engine.requests import (
    StartEngineTrainingRequest,
    PauseEngineTrainingRequest,
    ResumeEngineTrainingRequest,
    StopEngineTrainingRequest,
    EvaluateEngineCheckpointRequest,
    SaveEngineCheckpointRequest,
)

from training_service.application.dto.engine.responses import (
    EngineTrainingStartedResponse,
    EngineStatusResponse,
    EngineMetricsResponse,
    EngineCheckpointEvaluationResponse,
    EngineCheckpointSavedResponse,
)


class EngineClient(ABC):
    """
    Contract describing communication between
    training backend and engine
    """

        @abstractmethod
    def start(
        self,
        request: StartEngineTrainingRequest,
    ) -> EngineTrainingStartedResponse:
        raise NotImplementedError

    @abstractmethod
    def pause(
        self,
        request: PauseEngineTrainingRequest,
    ) -> EngineStatusResponse:
        raise NotImplementedError

    @abstractmethod
    def resume(
        self,
        request: ResumeEngineTrainingRequest,
    ) -> EngineStatusResponse:
        raise NotImplementedError

    @abstractmethod
    def stop(
        self,
        request: StopEngineTrainingRequest,
    ) -> EngineStatusResponse:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(
        self,
        request: SaveEngineCheckpointRequest,
    ) -> EngineCheckpointSavedResponse:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(
        self,
        request : LocdEngineCheckpointRequest
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_checkpoint(
        self,
        request : DeleteEngineCheckpointRequest,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate_checkpoint(
        self,
        request : EvaluateEngineCheckpointRequest,
    ) -> EngineCheckpointEvaluationResponse:
        raise NotImplementedError

    @abstractmethod
    def get_metrics(
        self,
        training_run_id,
    ) -> EngineMetricsResponse:
        raise NotImplementedError