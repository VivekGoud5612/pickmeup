from backend.engine_gateway.infrastructure.publishers.local_engine_event_publisher import (
    LocalEngineEventPublisher,
)

from backend.engine_gateway.application.events.training_events import *
from backend.engine_gateway.application.events.checkpoint_events import *
from backend.engine_gateway.application.events.curriculum_events import *
from backend.engine_gateway.application.events.evaluation_events import *

from backend.training_service.application.event_handlers.training_handlers import *
from backend.training_service.application.event_handlers.checkpoint_handlers import *
from backend.training_service.application.event_handlers.curriculum_handlers import *
from backend.training_service.application.event_handlers.evaluation_handlers import *


def create_local_engine_event_publisher(
    training_repo,
    configuration_repo,
    checkpoint_repo,
    evaluation_repo,
) -> LocalEngineEventPublisher:

    publisher = LocalEngineEventPublisher()

    # ---------------------------
    # Training
    # ---------------------------

    publisher.register_handler(
        TrainingInitializedEvent,
        TrainingInitializedHandler(training_repo),
    )

    publisher.register_handler(
        TrainingStartedEvent,
        TrainingStartedHandler(training_repo),
    )

    publisher.register_handler(
        TrainingPausedEvent,
        TrainingPausedHandler(training_repo),
    )

    publisher.register_handler(
        TrainingResumedEvent,
        TrainingResumedHandler(training_repo),
    )

    publisher.register_handler(
        TrainingStoppedEvent,
        TrainingStoppedHandler(training_repo),
    )

    publisher.register_handler(
        TrainingCompletedEvent,
        TrainingCompletedHandler(training_repo),
    )

    publisher.register_handler(
        TrainingFailedEvent,
        TrainingFailedHandler(training_repo),
    )

    # ---------------------------
    # Checkpoint
    # ---------------------------

    publisher.register_handler(
        CheckpointCreatedEvent,
        CheckpointCreatedHandler(
            training_repo,
            checkpoint_repo,
        ),
    )

    # ---------------------------
    # Curriculum
    # ---------------------------

    publisher.register_handler(
        CurriculumAdvancedEvent,
        CurriculumAdvancedHandler(
            training_repo,
            configuration_repo,
        ),
    )

    # ---------------------------
    # Evaluation
    # ---------------------------

    publisher.register_handler(
        EvaluationStartedEvent,
        EvaluationStartedHandler(),
    )

    publisher.register_handler(
        EvaluationCompletedEvent,
        EvaluationCompletedHandler(
            checkpoint_repo,
            evaluation_repo,
        ),
    )

    return publisher