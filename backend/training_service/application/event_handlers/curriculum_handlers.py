from __future__ import annotations

from backend.training_service.application.event_handlers.base import (
    EngineEventHandler,
)

from backend.engine_gateway.application.events.curriculum_events import (
    CurriculumAdvancedEvent,
)

from backend.training_service.application.repositories.training_repository import (
    TrainingRunRepository,
)

from backend.training_service.application.repositories.training_configuration_repository import (
    TrainingConfigurationRepository,
)


class CurriculumAdvancedHandler(EngineEventHandler):

    def __init__(
        self,
        training_repo: TrainingRunRepository,
        configuration_repo: TrainingConfigurationRepository,
    ) -> None:

        self._training_repo = training_repo
        self._configuration_repo = configuration_repo

    def handle(
        self,
        event: CurriculumAdvancedEvent,
    ) -> None:

        training = self._training_repo.get_by_id(
            event.run_id,
        )

        configuration = self._configuration_repo.get_by_id(
            training.configuration_id,
        )

        configuration.curriculum_settings = (
            configuration.curriculum_settings.with_difficulty(
                event.new_level,
            )
        )

        self._configuration_repo.update(
            configuration,
        )