from __future__ import annotations

from uuid import UUID

from engine.training.training_engine import (
    TrainingEngine,
)

from backend.engine_gateway.application.contracts.engine_client import (
    EngineClient,
)

from backend.engine_gateway.application.contracts.engine_event_publisher import (
    EngineEventPublisher,
)

from backend.engine_gateway.application.dto.requests import (
    InitializeEngineTrainingRequest,
)

from backend.engine_gateway.infrastructure.clients.local_engine_client import (
    LocalEngineClient,
)

from backend.engine_gateway.infrastructure.publisher_factory import (
    create_local_engine_event_publisher,
)

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)

from backend.training_service.application.repositories.training_configuration_repository import (
    TrainingConfigurationRepository,
)

from backend.training_service.application.repositories.checkpoint_repository import (
    CheckpointRepository,
)

from backend.training_service.application.repositories.evaluation_repository import (
    EvaluationRepository,
)


class EngineRegistry:

    def __init__(self) -> None:

        self._clients: dict[UUID, EngineClient] = {}

    def create(
        self,
        training_run_id: UUID,
        request: InitializeEngineTrainingRequest,

        training_repo: TrainingRunRepository,
        configuration_repo: TrainingConfigurationRepository,
        checkpoint_repo: CheckpointRepository,
        evaluation_repo: EvaluationRepository,

    ) -> EngineClient:

        if training_run_id in self._clients:
            raise ValueError(
                f"Engine already exists for run {training_run_id}"
            )

        publisher: EngineEventPublisher = (
            create_local_engine_event_publisher(
                training_repo=training_repo,
                configuration_repo=configuration_repo,
                checkpoint_repo=checkpoint_repo,
                evaluation_repo=evaluation_repo,
            )
        )

        engine = TrainingEngine()

        client = LocalEngineClient(
            engine=engine,
        )

        client.initialize(
            request=request,
            publisher=publisher,
        )

        print("=" * 60)
        print(f"CREATE REGISTRY ID : {id(self)}")
        print(f"REGISTERING RUN    : {training_run_id}")

        self._clients[training_run_id] = client

        print(f"CLIENTS AFTER CREATE: {list(self._clients.keys())}")
        print("=" * 60)

        return client

    def get(
        self,
        training_run_id: UUID,
    ) -> EngineClient:

        print("=" * 60)
        print(f"GET REGISTRY ID    : {id(self)}")
        print(f"LOOKING FOR        : {training_run_id}")
        print(f"AVAILABLE CLIENTS  : {list(self._clients.keys())}")
        print("=" * 60)

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

        self._clients.pop(
            training_run_id,
            None,
        )

    def exists(
        self,
        training_run_id: UUID,
    ) -> bool:

        return training_run_id in self._clients

    def running_training_runs(
        self,
    ) -> list[UUID]:

        return list(self._clients.keys())