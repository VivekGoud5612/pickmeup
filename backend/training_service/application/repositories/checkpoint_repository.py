from __future__ import annotations 

from abc import ABC 
from uuid import UUID 

from training_services.domain.entities.checkpoint import TrainingCheckpoint 

class CheckpointRepository(ABC):
    """
    Contract for persisting and retrieving training checkpoints.
    """

    @abstractmethod
    def save(self, checkpoint: TrainingCheckpoint,) -> None:
        """
        Persist a checkpoint.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, checkpoint: TrainingCheckpoint,) -> None:
        """
        Persist changes made to a checkpoint.
        """
        raise NotImplementedError

    @abstractmethod
    def get_by_id(self, checkpoint_id: UUID,) -> TrainingCheckpoint:
        """
        Retrieve a checkpoint by its identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def get_latest_for_run(self, training_run_id: UUID,) -> TrainingCheckpoint | None:
        """
        Retrieve the most recently created checkpoint for a training run.
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_for_run(self, training_run_id: UUID,) -> TrainingCheckpoint | None:   
        """
        Retrieve the best checkpoint for a training run.
        """
        raise NotImplementedError

    @abstractmethod
    def list_by_run(self, training_run_id: UUID,) -> list[TrainingCheckpoint]:  ## AS we store latest checkpoint in run as well...
        """
        Retrieve every checkpoint belonging to a training run.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, checkpoint_id: UUID,) -> None:
        """
        Delete a checkpoint.
        """
        raise NotImplementedError