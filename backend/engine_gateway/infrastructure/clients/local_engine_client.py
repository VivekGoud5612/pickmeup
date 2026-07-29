from __future__ import annotations
from uuid import UUID 
from datetime import datetime, UTC 

from backend.training_service.domain.enums import CheckpointType

from engine.training.training_engine import (
    TrainingEngine,
)

from engine_gateway.application.contracts.engine_client import (
    EngineClient,
)

from engine_gateway.application.contracts.engine_event_publisher import (
    EngineEventPublisher,
)

from training_service.application.dto.engine.requests import (
    StartEngineTrainingRequest,
    PauseEngineTrainingRequest,
    ResumeEngineTrainingRequest,
    StopEngineTrainingRequest,
    EvaluateEngineCheckpointRequest,
    SaveEngineCheckpointRequest,
    DeleteEngineCheckpointRequest,
)

from training_service.application.dto.engine.responses import (
    EngineTrainingStartedResponse,
    EngineStatusResponse,
    EngineMetricsResponse,
    EngineCheckpointEvaluationResponse,
    EngineCheckpointSavedResponse,
)


class LocalEngineClient(EngineClient):
    """
    Local adapter - takes in engine client contract to have the exact same functionalities
    Directly talks with engine, and the backend actually talks with this

    So engine remains safe, and this provides some functionalities for the use case to
    read or write to the engine specific metrics
    """

    def __init__(self, engine : TrainingEngine) -> None:
        self._engine = engine 


    def initialize(self, request : InitializeEngineTrainingRequest, publisher : EngineEventPublisher) -> None:
        """
        Initialize the engine for a new training session

        Also thought of publisher comes from backend, but no need .. and thought of defining the publisher
        in the constructor, but publisher is perisisted to the engine and not the client so to avoid duplication
        """
        self._engine.initializate(
            config = request.configuration,
            run_name = request.run_name,
            publisher = publisher,
        )


    def start(self, request : StartEngineTrainingRequest) -> EngineTrainingStartedResponse:
        self._engine.start()

        return EngineTrainingStartedResponse(
            status = self._engine.state,
        )

    def pause(self) -> EngineStatusResponse:
    """
    Pause the current training session.
    """
        self._engine.pause()

        return EngineStatusResponse(
            status = self._engine.state,
        )


    def stop(self) -> EngineStatusResponse:
    """
    Stop training.
    """
        self._engine.stop()

        return EngineStatusResponse(
            status = self._engine.state,
        )
    

    def resume(self) -> EngineStatusResponse:
    """
    Resume a paused training session.
    """
        self._engine.resume()

        return EngineStatusResponse(
            status = self._engine.state,
        )

    
    def save_checkpoint(
    self,
    request: SaveEngineCheckpointRequest,
    ) -> EngineCheckpointSavedResponse:
    """
    Save the current engine state.
    """

        checkpoint_path = (
            self._engine.checkpoint_directory()/  ## A read only property which returns the same...
            request.checkpoint_name/
            f"step_{self._engine.global_step()}.pt"
        )

        self._engine.save_checkpoint(checkpoint_path)

        return EngineCheckpointResponse{
            checkpoint_type = CheckpointType.MANUAL,
            checkpoint_path = checkpoint_path,   ## Is thesame response for periodic checkpoint as well, but need to make it so that each time it gets created, we get some signal from the engine to send to backend, that the checkpoint was created and 
            ## you need to update that an dsave that, and update training run to contain that as the latest checkpoint...
            created_at = datetime.now(UTC),
        }

    
    def load_checkpoint(
    self,
    request: LoadEngineCheckpointRequest,
    ) -> None:
    """
    Restore a previous checkpoint.
    """

        self._engine.load_checkpoint(
            request.path,
        )


    def delete_checkpoint(
        self, 
        request : DeleteEngineCheckpointRequest,
    ) -> None:
        
        if request.checkpoint_path.exists():  ### Note that we get Path objects from DTOs, but are just simply strings favourible ?? 
    
    def evaluate_checkpoint(self, request : EvaluateEngineCheckpointRequest) -> EngineCheckpointEvaluationResponse:
        """
        Nothing much for now, just basic get metric function
        """

        metrics = self._engine.evaluate(request.checkpoint_path)

        return EngineCheckpointEvaluationResponse(
            checkpoint_path = request.checkpoint_path,
            metrics = metrics,
        )