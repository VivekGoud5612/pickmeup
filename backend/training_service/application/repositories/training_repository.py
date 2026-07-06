from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from training_services.domain.entities.training_run import TrainingRun


class TrainingRunRepository(ABC):
    """
    Contract for persisting and retrieving training runs.

    The application layer depends on this abstraction instead
    of a concrete database implementation.

    That is everything is requested for from here and not directly to 
    the databse
    """

    @abstractmethod 
    def save(self, training_run : TrainingRun,) -> None:
        """
        Persist a new training run
        """
        raise NotImplementedError

    @abstractmethod 
    def update(self, training_run : TrainingRun,) -> None:
        """
        Persist changes made to an existing training run
        """
        raise NotImplementedError 

    @abstractmethod 
    def get_by_id(self, training_run_id : UUID) -> TrainingRun:
        """
        Retrieve a training run by its identifier.
        """
        raise NotImplementedError 

    @abstractmethod 
    def delete(self, training_run_id: UUID) -> None:
        """
        Delete that specific traning run
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, training_run_id : UUID) -> bool:
        """
        check if the training run exists ??
        """
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> list[TrainingRun]:
        """
        Retrieve all training runs
        """
        raise NotImplementedError
