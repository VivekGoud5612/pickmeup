from __future__ import annotations 

from abc import ABC, abstractmethod 

from training_service.application.dto.engine.requests import (
    StartEngineTrainingRequest,
    PauseEngineTrainingRequest,
    ResumeEngineTrainingRequest,
    StopEngineTrainingRequest,
    EvaluateCheckpointRequest,
    SaveCheckpointRequest,
)

from training_service.application.dto.engine.responses import (
    EngineTrainingStartedResponse,
    EngineStatusResponse,
    EngineMetricsResponse,
    EngineEvaluationResponse,
    EngineCheckpointResponse,
)


class EngineClient(ABC):
    """
    Contract describing communication between
    training backend and engine
    """

        @abstractmethod
    def start_training(
        self,
        request: StartEngineTrainingRequest,
    ) -> EngineTrainingStartedResponse:
        raise NotImplementedError

    @abstractmethod
    def pause_training(
        self,
        request: PauseEngineTrainingRequest,
    ) -> EngineStatusResponse:
        raise NotImplementedError

    @abstractmethod
    def resume_training(
        self,
        request: ResumeEngineTrainingRequest,
    ) -> EngineStatusResponse:
        raise NotImplementedError

    @abstractmethod
    def stop_training(
        self,
        request: StopEngineTrainingRequest,
    ) -> EngineStatusResponse:
        raise NotImplementedError

    @abstractmethod
    def evaluate_checkpoint(
        self,
        request: EvaluateCheckpointRequest,
    ) -> EngineEvaluationResponse:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(
        self,
        request: SaveCheckpointRequest,
    ) -> EngineCheckpointResponse:
        raise NotImplementedError

    @abstractmethod
    def get_metrics(
        self,
        training_run_id,
    ) -> EngineMetricsResponse:
        raise NotImplementedError