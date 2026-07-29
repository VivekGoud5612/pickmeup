from __future__ import annotations 
from uuid import UUID 

from engine_gateway.application.engine_client import (
    EngineClient,
)

from engine_gateway.infrastructure.clients.local_engine_client import (
    LocalEngineClient,
)

from engine_gateway.application.contracts.engine_event_publisher import (
    EngineEventPublisher,
)

from training_service.application.dto.engine.requests import (
    InitializeEngineTrainingRequest,
)

from engine.training.training_engine import TrainingEngine



class EngineRegistry:

    def __init__(self) -> None:
        self._clients : dict[UUID, EngineClient] = {}


    def create(
        self, 
        training_run_id : UUID,
        request : InitializeEngineTrainingRequest,
        publisher : EngineEventPublisher,
    ) -> EngineClient:

        if training_run_id in self._clients:
            raise ValueError(f"Engine already exists for run {training_run_id}")

        engine = TrainingEngine()

        client = LocalEngineClient(engine = engine)

        client.initialize(
            request = request,
            publisher = publisher,
        )

        self._clients[training_run_id] = client 

        return client 


    def get(
        self,
        training_run_id: UUID,
    ) -> EngineClient:

        try:
            return self._clients[training_run_id]

        except KeyError:
            raise ValueError(
                f"No engine registered for run {training_run_id}"
            )

    def remove(
        self,
        training_run_id: UUID,
    ) -> None:

        self._clients.pop(training_run_id, None)

    def exists(
        self,
        training_run_id: UUID,
    ) -> bool:

        return training_run_id in self._clients

    def running_training_runs(
        self,
    ) -> list[UUID]:

        return list(self._clients.keys())