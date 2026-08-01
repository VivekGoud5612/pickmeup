from __future__ import annotations

from abc import ABC, abstractmethod

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


class EngineClient(ABC):
    """
    Contract for communicating with a single initialized
    training engine instance.

    One EngineClient <-> One TrainingEngine.
    """

    @abstractmethod
    def initialize(
        self,
        request: InitializeEngineTrainingRequest,
    ) -> None:
        """
        Creates and initializes the underlying training engine.
        Must be called exactly once before start().
        """
        raise NotImplementedError

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
    def delete_checkpoint(
        self,
        request: DeleteEngineCheckpointRequest,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate_checkpoint(
        self,
        request: EvaluateEngineCheckpointRequest,
    ) -> EngineCheckpointEvaluationResponse:
        raise NotImplementedError

    @abstractmethod
    def get_metrics(
        self,
    ) -> EngineMetricsResponse:
        """
        Returns the latest metrics from the running engine.
        """
        raise NotImplementedError

    @abstractmethod
    def replay(
        self,
        request: ReplayCheckpointRequest,
    ) -> None:
        """
        Runs inference on a checkpoint and streams frames
        to the frontend.
        """
        raise NotImplementedError