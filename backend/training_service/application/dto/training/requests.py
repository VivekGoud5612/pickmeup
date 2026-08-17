"""
Request DTOs for Training use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from backend.training_service.application.dto.training.nested_requests import (
    CurriculumSettingsRequest,
    HyperParametersRequest,
    RewardWeightsRequest,
)
from backend.training_service.domain.enums import TrainingAlgorithm, TrainingStatus


@dataclass(slots=True, frozen=True, kw_only=True)
class StartTrainingRequest:
    name: str
    configuration_name: str
    algorithm: TrainingAlgorithm
    hyperparameters: HyperParametersRequest
    reward_weights: RewardWeightsRequest
    curriculum_settings: CurriculumSettingsRequest
    notes: str = ""


@dataclass(slots=True, frozen=True, kw_only=True)
class PauseTrainingRequest:
    training_run_id: UUID


@dataclass(slots=True, frozen=True, kw_only=True)
class ResumeTrainingRequest:
    training_run_id: UUID


@dataclass(slots=True, frozen=True, kw_only=True)
class StopTrainingRequest:
    training_run_id: UUID


@dataclass(slots=True, frozen=True, kw_only=True)
class FailTrainingRequest:
    training_run_id: UUID


@dataclass(slots=True, frozen=True, kw_only=True)
class DeleteTrainingRequest:
    training_run_id: UUID


@dataclass(slots=True, frozen=True, kw_only=True)
class GetTrainingSummaryRequest:
    training_run_id: UUID


@dataclass(slots=True, frozen=True, kw_only=True)
class GetTrainingProgressRequest:
    training_run_id: UUID


@dataclass(slots=True, frozen=True, kw_only=True)
class ListTrainingRunsRequest:
    status: TrainingStatus | None = None
    algorithm: TrainingAlgorithm | None = None
    limit: int = 100
