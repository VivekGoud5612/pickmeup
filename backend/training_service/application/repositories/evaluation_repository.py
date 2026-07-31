from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from backend.training_service.domain.entities.evaluation_result import (
    EvaluationResult,
)


class EvaluationRepository(ABC):
    """
    Contract for persisting and retrieving evaluation results.
    """

    @abstractmethod
    def save(
        self,
        evaluation: EvaluationResult,
    ) -> None:
        """
        Persist an evaluation result.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        evaluation: EvaluationResult,
    ) -> None:
        """
        Persist changes made to an evaluation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_by_id(
        self,
        evaluation_id: UUID,
    ) -> EvaluationResult:
        """
        Retrieve an evaluation by its identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def list_by_checkpoint(
        self,
        checkpoint_id: UUID,
    ) -> list[EvaluationResult]:
        """
        Retrieve all evaluations for a checkpoint.
        """
        raise NotImplementedError

    @abstractmethod
    def get_latest_by_checkpoint(
        self,
        checkpoint_id: UUID,
    ) -> EvaluationResult | None:
        """
        Retrieve the latest evaluation for a checkpoint.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        evaluation_id: UUID,
    ) -> None:
        """
        Delete an evaluation result.
        """
        raise NotImplementedError