from __future__ import annotations

from uuid import UUID

from backend.contracts.engine.dto.requests import InitializeEngineTrainingRequest
from backend.engine_gateway.application.contracts.engine_client import EngineClient
from backend.engine_gateway.application.contracts.engine_event_publisher import (
    EngineEventPublisher,
)
from backend.engine_gateway.infrastructure.clients.local_engine_client import (
    LocalEngineClient,
)
from backend.engine_service.application.engine_service import LocalEngineService


class EngineClientRegistry:
    """Local composition helper for scoped SDK clients.

    Runtime workers remain in ``LocalEngineService``; this object only retains
    clients for the current in-process composition root.
    """
    def __init__(self) -> None:
        self._clients: dict[UUID, EngineClient] = {}
        self._service = LocalEngineService()

    def create(
        self,
        training_run_id: UUID,
        request: InitializeEngineTrainingRequest,
        publisher: EngineEventPublisher,
    ) -> EngineClient:
        if training_run_id in self._clients:
            raise ValueError(f"Engine already exists for run {training_run_id}")

        client = LocalEngineClient(service=self._service, training_run_id=training_run_id)
        client.initialize(request=request, publisher=publisher)
        self._clients[training_run_id] = client
        return client

    def get(self, training_run_id: UUID) -> EngineClient:
        try:
            return self._clients[training_run_id]
        except KeyError as exc:
            raise ValueError(f"No engine registered for run {training_run_id}") from exc

    def remove(self, training_run_id: UUID) -> None:
        self._clients.pop(training_run_id, None)
        self._service.remove(training_run_id)

    def exists(self, training_run_id: UUID) -> bool:
        return training_run_id in self._clients

    def running_training_runs(self) -> list[UUID]:
        return list(self._clients.keys())


# Temporary source-compatible name while composition roots migrate.
EngineRegistry = EngineClientRegistry
