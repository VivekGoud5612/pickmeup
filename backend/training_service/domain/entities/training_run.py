from __future__ import annotations

from dataclasses import dataclass, field 
from datetime import datetime, UTC
from uuid import UUID, uuid4

from backend.training_service.domain.enums import TrainingAlgorithm, TrainingStatus
from backend.training_service.domain.value_objects import TrainingProgress

@dataclass(slots = True, kw_only = True)  ## Refer dataclass notes..
class TrainingRun:
    """
    Represents a single training session of the algorithm   
    This entity tracks only lifecycle and progress of training, metrics and checkpoints
    and recommendations are maintained by their own domain entities
    """

    id : UUID = field(default_factory = uuid4)
        
    name : str 
    algorithm : TrainingAlgorithm

    status : TrainingStatus = TrainingStatus.CREATED 

    progress : TrainingProgress | None = None  ## Remove current step, current episode... and move them into Training Progress which is a value object wiht no identity

    configuration_id : UUID | None = None   ## Each training session or epsidoe could have its configuration of gamma, rewards and such
    latest_checkpoint_id : UUID | None = None  ## Instead of storing the whole checkpoint object which is latest, we cimply store the id of that latest as checkpoint si also an entity with different objects means different things
    best_checkpoint_id : UUID | None = None

    started_at : datetime = field(default_factory = lambda : datetime.now(UTC)) 
    finished_at : datetime | None = None 

    notes : str = ""  ## MAybe to store some information regarding this run.. Not the report but some meta information...


    def start(self) -> None:
        if self.status != TrainingStatus.CREATED:
            raise RuntimeError("Training has already been started")  ## Because the first time we start the status needs to be created indicating the trainingrn object has just been created..

        self.status = TrainingStatus.RUNNING 
        self.started_at = datetime.now(UTC)

    def pause(self) -> None:
        if self.status != TrainingStatus.RUNNING:
            raise RuntimeError("Training is not being run")
            
        self.status = TrainingStatus.PAUSED 

    def resume(self) -> None:
        if self.status != TrainingStatus.PAUSED:
            raise RuntimeError("Training is not paused")
            
        self.status = TrainingStatus.RUNNING 

    def stop(self) -> None:
        if self.status not in (TrainingStatus.RUNNING, TrainingStatus.PAUSED):
            raise RuntimeError("Training is not running or is not paused, so it cannot be stopped")

        self.status = TrainingStatus.COMPLETED
        self.finished_at = datetime.now(UTC)

    def fail(self) -> None:
        self.status = TrainingStatus.FAILED
        self.finished_at = datetime.now(UTC)

        
    def attach_latest_checkpoint(self, checkpoint_id : UUID) -> None:   ## attach object to object...
        self.latest_checkpoint_id = checkpoint_id   ## Think of TrainingCheckpointConfig as a entity containing things like id, time and suhc.

    def attach_best_checkpoint(self, checkpoint_id : UUID) -> None:   ## attach object to object...
        self.best_checkpoint_id = checkpoint_id 

    def attach_config(self, config_id : UUID) -> None:
        self.configuration_id = config_id 

    def detach_latest_checkpoint(self) -> None:
        self.latest_checkpoint_id = None