from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from backend.engine_gateway.application.events.base import (
    EngineEvent,
)

from backend.training_service.domain.enums import (
    CheckpointType,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class CheckpointCreatedEvent(EngineEvent):

    run_id: UUID

    checkpoint_path: Path

    checkpoint_type: CheckpointType

    global_step: int

    hyperparameters: HyperParameters

    reward_weights: RewardWeights

    curriculum_settings: CurriculumSettings

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )