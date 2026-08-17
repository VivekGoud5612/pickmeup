"""Application facade for the local Engine Service implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from backend.contracts.engine.dto.requests import InitializeEngineTrainingRequest
from backend.contracts.engine.enums import EngineStatus
from backend.contracts.engine.event_publisher import EngineEventPublisher
from backend.engine_service.application.runtime_registry import RuntimeRegistry
from backend.engine_service.domain.entities.runtime_worker import RuntimeWorker
if TYPE_CHECKING:
    from engine.training.training_engine import TrainingEngine


class LocalEngineService:
    """Executes local workloads while keeping runtime ownership in Engine.

    This is a service implementation, not a gateway adapter.  HTTP/gRPC can
    expose the same operations later without changing the runtime model.
    """

    def __init__(self, registry: RuntimeRegistry | None = None) -> None:
        self._registry = registry or RuntimeRegistry()

    def initialize_training(
        self, request: InitializeEngineTrainingRequest, publisher: EngineEventPublisher
    ) -> None:
        # The algorithm/runtime dependency is loaded only when this local
        # execution implementation is actually selected.  Control-plane and
        # gateway imports therefore do not require Torch.
        from engine.training.training_engine import TrainingEngine

        engine = TrainingEngine()
        engine.initialize(
            config=request.configuration,
            run_id=request.run_id,
            run_name=request.run_name,
            checkpoint_directory=request.checkpoint_directory,
            publisher=publisher,
        )
        self._registry.add(
            RuntimeWorker(
                run_id=request.run_id,
                execution_handle=engine,
                status=EngineStatus.INITIALIZED,
            )
        )

    def engine_for(self, run_id: UUID) -> "TrainingEngine":
        worker = self._registry.get(run_id)
        return worker.execution_handle

    def worker_for(self, run_id: UUID) -> RuntimeWorker:
        return self._registry.get(run_id)

    def remove(self, run_id: UUID) -> None:
        self._registry.remove(run_id)
