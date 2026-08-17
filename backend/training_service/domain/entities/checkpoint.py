from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from backend.contracts.engine.enums import CheckpointType
from backend.contracts.engine.models.training_configuration import (
    CurriculumSettings,
    HyperParameters,
    RewardWeights,
)
from backend.contracts.engine.models.training_progress import TrainingProgress


@dataclass(slots=True, kw_only=True)
class TrainingCheckpoint:
    training_run_id: UUID
    progress: TrainingProgress
    checkpoint_type: CheckpointType
    file_path: Path
    hyperparameters: HyperParameters
    reward_weights: RewardWeights
    curriculum_settings: CurriculumSettings
    id: UUID = field(default_factory=uuid4)
    is_best: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    description: str = ""

    def mark_as_best(self) -> None:
        self.is_best = True

    def rename(self, description: str) -> None:
        self.description = description
