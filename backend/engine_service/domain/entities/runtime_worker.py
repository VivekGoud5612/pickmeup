from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC 
from uuid import UUID 

from engine.utils.enums import (
    EngineStatus,
)
from domain.value_objects import CheckpointMetaData


@dataclass(slots = True, kw_only = True)
class RuntimeWorker:

    training_run_id : UUID 

    status : EngineStatus

    started_at : datetime

    finished_at : datetime | None = None 

    latest_checkpoint : CheckpointMetaData | None   ## A run time state , but useful when checked for status

    def start(self) -> None:
        if self.status != EngineStatus.CREATED:
            raise RuntimeError("Training has already been started")  ## Because the first time we start the status needs to be created indicating the trainingrn object has just been created..

        self.status = EngineStatus.RUNNING 
        self.started_at = datetime.now(UTC)

    def pause(self) -> None:
        if self.status != EngineStatus.RUNNING:
            raise RuntimeError("Training is not being run")
        
        self.status = EngineStatus.PAUSED 

    def resume(self) -> None:
        if self.status != EngineStatus.PAUSED:
            raise RuntimeError("Training is not paused")
        
        self.status = EngineStatus.RUNNING 

    def stop(self) -> None:
        if self.status not in (EngineStatus.RUNNING, EngineStatus.PAUSED):
            raise RuntimeError("Training is not running or is not paused, so it cannot be stopped")

        self.status = EngineStatus.STOPPED
        self.finished_at = datetime.now(UTC)
